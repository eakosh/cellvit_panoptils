import yaml
import torch
import argparse
import wandb
import os
from pathlib import Path

from model.cellvit import CellViT
from datasets.panoptils import PanopTILsDataset, PanopTILsPaths
from data.splits import load_splits
from data.datamodule import DataConfig, PanopTILsDataModule
from data.transforms import create_train_transforms, create_val_transforms
from training.losses import CellViTMultiTaskLoss
from training.trainer import Trainer


def load_vit_dino_pretrained(model, ckpt_path):
    print(f"Loading DINO ViT weights from {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "teacher" in checkpoint:
        print("Found DINO checkpoint with 'teacher' key")
        state = checkpoint["teacher"]
    elif "student" in checkpoint:
        print("Found DINO checkpoint with 'student' key")
        state = checkpoint["student"]
    elif "state_dict" in checkpoint:
        print("Found checkpoint with 'state_dict' key")
        state = checkpoint["state_dict"]
    else:
        state = checkpoint

    new_state = {}
    for k, v in state.items():
        k = k.replace("module.", "")
        k = k.replace("backbone.", "")
        new_state[k] = v

    missing, unexpected = model.encoder.load_state_dict(
        new_state,
        strict=False
    )

    print("DINO encoder loaded")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    if len(missing) > 10:
        print(f"WARNING: {len(missing)} keys are missing. Encoder may not be properly initialized")
        print(f"\tFirst 5 missing: {missing[:5]}")
    else:
        print("Encoder weights loaded successfully")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="Override dataset root path")
    parser.add_argument("--encoder-path", type=str, default=None,
                        help="Override encoder checkpoint path")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from checkpoint")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.dataset_path is not None:
        cfg["data"]["root"] = args.dataset_path
        print(f"Using dataset path from args: {args.dataset_path}")

    if args.encoder_path is not None:
        cfg["model"]["encoder_pretrained"] = args.encoder_path
        print(f"Using encoder path from args: {args.encoder_path}")


    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"\tAvailable memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        # A100: enable TF32 for faster matrix multiplications
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    else:
        device = "cpu"
        print("GPU not detected, using CPU")

    # output directory for checkpoints
    output_dir = Path("outputs") / f"fold_{cfg['splits']['fold']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved to: {output_dir}")

    cfg["data"]["unlabeled_class"] = cfg["model"].get("nuclei_unlabeled_class", None)
    cfg["data"]["background_nuclei_class"] = cfg["model"].get("nuclei_background_class", None)

    paths = PanopTILsPaths(root=cfg["data"]["root"], subset=cfg["data"]["subset"])
    tmp_ds = PanopTILsDataset(paths=paths, file_list=None, transforms=None, cache_dataset=False, include_tissue_label=False)
    all_files = tmp_ds.files

    fold_id = cfg["splits"]["fold"]
    train_files, val_files = load_splits(all_files, root=cfg["data"]["root"], fold=fold_id)

    print(f"\nDataset: {cfg['data']['subset']}")
    print(f"Total files: {len(all_files)}")
    print(f"Fold {fold_id}/5: Train={len(train_files)}, Val={len(val_files)}\n")

    # data augmentation
    train_transforms = create_train_transforms(image_size=256)
    val_transforms = create_val_transforms(image_size=256)

    dm = PanopTILsDataModule(
        cfg=DataConfig(**cfg["data"]),
        train_files=train_files,
        val_files=val_files,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
    )
    dm.setup()

    model = CellViT(
        num_nuclei_classes=cfg["model"]["num_nuclei_classes"],
        num_tissue_classes=cfg["model"]["num_tissue_classes"],
        embed_dim=cfg["model"]["embed_dim"],
        input_channels=3,
        depth=cfg["model"]["depth"],
        num_heads=cfg["model"]["num_heads"],
        extract_layers=cfg["model"]["extract_layers"],
    )

    # load DINO pretrained weights
    ckpt = cfg["model"].get("encoder_pretrained", None)
    if ckpt is not None and os.path.exists(ckpt):
        load_vit_dino_pretrained(model, ckpt)
    else:
        print(f"Warning: Encoder checkpoint not found at {ckpt}. Training from scratch")

    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="max-autotune")
            print("Model compiled with torch.compile(mode='max-autotune')")
        except Exception as e:
            print(f"torch.compile not available, continuing without: {e}")

    unlabeled_class = cfg["model"].get("nuclei_unlabeled_class", None)
    background_class = cfg["model"].get("nuclei_background_class", None)

    loss_cfg = cfg["loss"]
    loss_fn = CellViTMultiTaskLoss(
        lambda_np_ft=loss_cfg.get("lambda_np_ft", 1.0),
        lambda_np_dice=loss_cfg.get("lambda_np_dice", 1.0),
        lambda_hv_mse=loss_cfg.get("lambda_hv_mse", 2.5),
        lambda_hv_msge=loss_cfg.get("lambda_hv_msge", 8.0),
        lambda_nt_ft=loss_cfg.get("lambda_nt_ft", 0.5),
        lambda_nt_dice=loss_cfg.get("lambda_nt_dice", 0.2),
        lambda_nt_bce=loss_cfg.get("lambda_nt_bce", 0.5),
        lambda_tc_ce=loss_cfg.get("lambda_tc_ce", 0.1),
        ft_alpha=loss_cfg.get("ft_alpha", 0.7),
        ft_beta=loss_cfg.get("ft_beta", 0.3),
        ft_gamma=loss_cfg.get("ft_gamma", 4.0 / 3.0),
        ft_eps=loss_cfg.get("ft_eps", 1e-6),
        unlabeled_class=unlabeled_class,
    )
    
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["wd"],
        betas=cfg["train"].get("betas", (0.85, 0.95))
    )
    
    scheduler_type = cfg["train"].get("scheduler_type", "exponential")
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim,
            T_max=cfg["train"]["epochs"],
            eta_min=cfg["train"].get("scheduler_eta_min", 1e-5),
        )
        print(f"Using CosineAnnealingLR scheduler (eta_min={cfg['train'].get('scheduler_eta_min', 1e-5)})")
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optim,
            gamma=cfg["train"].get("scheduler_gamma", 0.85),
        )
        print(f"Using ExponentialLR scheduler (gamma={cfg['train'].get('scheduler_gamma', 0.85)})")

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optim,
        scheduler=scheduler,
        device=device,
        use_mixed_precision=cfg["train"].get("use_mixed_precision", True),
        gradient_accumulation_steps=cfg["train"].get("gradient_accumulation_steps", 1),
        freeze_encoder_epochs=cfg["train"].get("freeze_encoder_epochs", 25),
        max_grad_norm=cfg["train"].get("max_grad_norm", 1.0),
        early_stopping_patience=cfg["train"].get("early_stopping_patience", None),
        num_nuclei_classes=cfg["model"]["num_nuclei_classes"],
        unlabeled_class=unlabeled_class,
        background_class=background_class,
    )

    start_epoch = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        start_epoch = trainer.load_checkpoint(args.checkpoint)
        print(f"Resuming from epoch {start_epoch}")

    wandb.init(
        project="cellvit-panoptils",
        config={
            "learning_rate": cfg["train"]["lr"],
            "weight_decay": cfg["train"]["wd"],
            "betas": cfg["train"].get("betas", (0.85, 0.95)),
            "epochs": cfg["train"]["epochs"],
            "batch_size": cfg["data"]["batch_size"],
            "gradient_accumulation_steps": cfg["train"].get("gradient_accumulation_steps", 1),
            "scheduler_gamma": cfg["train"].get("scheduler_gamma", 0.85),
            "freeze_encoder_epochs": cfg["train"].get("freeze_encoder_epochs", 25),
            "use_mixed_precision": cfg["train"].get("use_mixed_precision", True),
            "use_weighted_sampler": cfg["data"]["use_weighted_sampler"],
            "gamma_s": cfg["data"].get("gamma_s", 0.85),
            "fold": cfg["splits"]["fold"],
            "model": cfg["model"],
            "loss_weights": cfg["loss"],
        },
        name=f"fold_{cfg['splits']['fold']}",
    )

    best_pq = 0.0
    best_epoch = 0

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    print(f"\nStarting training for {cfg['train']['epochs']} epochs")
    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Val batches per epoch: {len(val_loader)}\n")

    val_metric_interval = cfg["train"].get("val_metric_interval", 5)
    print(f"Full validation metrics every {val_metric_interval} epochs")

    log_image_interval = cfg["train"].get("log_image_interval", 10)
    print(f"Prediction images logged to wandb every {log_image_interval} epochs")

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        print(f"\nEpoch {epoch}/{cfg['train']['epochs']}")
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")

        # train
        train_metrics = trainer.train_epoch(train_loader, epoch=epoch)

        # validate
        is_last_epoch = (epoch == cfg["train"]["epochs"] - 1)
        compute_full = (epoch % val_metric_interval == 0) or is_last_epoch
        val_metrics = trainer.val_epoch(val_loader, epoch=epoch, compute_full_metrics=compute_full)

        scheduler.step()

        metrics = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "train/loss": train_metrics['loss'],
            "val/loss": val_metrics['loss'],
        }

        for key, value in train_metrics.items():
            if key != 'loss':
                metrics[f"train/{key}"] = value

        for key, value in val_metrics.items():
            if key != 'loss':
                metrics[f"val/{key}"] = value

        if device == "cuda":
            mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
            metrics["gpu/memory_allocated_gb"] = mem_allocated
            metrics["gpu/memory_reserved_gb"] = mem_reserved

        if epoch % log_image_interval == 0:
            metrics.update(trainer.log_prediction_images(val_loader))

        wandb.log(metrics)

        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        if 'pq' in val_metrics:
            print(f"  Val PQ: {val_metrics['pq']:.4f}")
            print(f"  Val F1: {val_metrics['f1']:.4f}")

        latest_path = output_dir / "latest_checkpoint.pth"
        trainer.save_checkpoint(str(latest_path), epoch, val_metrics)

        if 'pq' in val_metrics and val_metrics['pq'] > best_pq:
            best_pq = val_metrics['pq']
            best_epoch = epoch
            checkpoint_path = output_dir / "best_model.pth"
            trainer.save_checkpoint(str(checkpoint_path), epoch, val_metrics)
            print(f"  New best model with PQ: {best_pq:.4f}")

        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pth"
            trainer.save_checkpoint(str(checkpoint_path), epoch, val_metrics)

        if trainer.check_early_stopping(val_metrics):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    print(f"\nTraining completed")
    print(f"Best PQ: {best_pq:.4f} at epoch {best_epoch}\n")

    wandb.finish()


if __name__ == "__main__":
    main()
