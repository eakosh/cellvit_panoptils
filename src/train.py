# Copyright © 2025 Olena Kosharova, FIIT STU.
# Part of the "Tissue-Context CellViT Extension" (bachelor's thesis, FIIT STU).
# Licensed under the Apache License 2.0 with the Commons Clause restriction.
# See the LICENSE file in the project root for full terms.

import yaml
import torch
import argparse
import wandb
import os
import uuid
from pathlib import Path

from model.cellvit import CellViT
from model.cellvit_panoptils import CellViTWithTissue
from datasets.panoptils import PanopTILsDataset, PanopTILsPaths
from data.splits import load_splits, load_dev_split
from data.datamodule import DataConfig, PanopTILsDataModule
from data.constants import NUCLEI_TISSUE_COMPATIBILITY
from data.transforms import create_train_transforms, create_val_transforms
from training.losses import CellViTMultiTaskLoss, CellViTTissueLoss
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

    missing, unexpected = model.encoder.load_state_dict(new_state, strict=False)

    print("DINO encoder loaded")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    if len(missing) > 10:
        print(f"Warning: {len(missing)} keys are missing. Encoder may not be properly initialized")
        print(f"\tFirst 5 missing: {missing[:5]}")
    else:
        print("Encoder weights loaded successfully")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="Override dataset root path")
    parser.add_argument("--dataset-subdir", type=str, default=None,
                        help="Subdirectory inside --dataset-path")
    parser.add_argument("--encoder-path", type=str, default=None,
                        help="Override encoder checkpoint path")
    parser.add_argument("--encoder-filename", type=str, default=None,
                        help="Encoder weights filename inside --encoder-path")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--checkpoint-mount", type=str, default=None,
                        help="Persistent blob mount for checkpoints")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory for checkpoints")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Unique run ID for auto-resume")
    parser.add_argument("--no-strict", action="store_true",
                        help="Load checkpoint with strict=False")
    parser.add_argument("--gamma-s", type=float, default=None,
                        help="Override data.gamma_s")
    parser.add_argument("--nt-weight-scale", type=float, default=None,
                        help="Multiply lambda_nt_ft/dice/bce by this scale")
    parser.add_argument("--fusion-embed-dim", type=int, default=None,
                        help="Override model.fusion_embed_dim (cross_attn_bottleneck)")
    parser.add_argument("--fusion-reduction", type=int, default=None,
                        help="Override model.fusion_reduction (AFF channel attention reduction)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.gamma_s is not None:
        cfg["data"]["gamma_s"] = args.gamma_s
        print(f"Override: data.gamma_s = {args.gamma_s}")
    if args.nt_weight_scale is not None:
        scale = args.nt_weight_scale
        for key in ("lambda_nt_ft", "lambda_nt_dice", "lambda_nt_bce"):
            base = cfg["loss"].get(key, 0.0)
            cfg["loss"][key] = round(base * scale, 6)
        print(f"Override: NT loss weights with {scale}: ft={cfg['loss']['lambda_nt_ft']},"
              f"dice={cfg['loss']['lambda_nt_dice']}, bce={cfg['loss']['lambda_nt_bce']}")
    if args.fusion_embed_dim is not None:
        cfg["model"]["fusion_embed_dim"] = args.fusion_embed_dim
        print(f"Override: model.fusion_embed_dim = {args.fusion_embed_dim}")
    if args.fusion_reduction is not None:
        cfg["model"]["fusion_reduction"] = args.fusion_reduction
        print(f"Override: model.fusion_reduction = {args.fusion_reduction}")

    if args.dataset_path is not None:
        dataset_root = args.dataset_path
        if args.dataset_subdir:
            dataset_root = os.path.join(dataset_root, args.dataset_subdir)
        cfg["data"]["root"] = dataset_root
        print(f"Using dataset path from args: {dataset_root}")

    if args.encoder_path is not None:
        encoder_path = args.encoder_path
        if args.encoder_filename:
            encoder_path = os.path.join(encoder_path, args.encoder_filename)
        cfg["model"]["encoder_pretrained"] = encoder_path
        print(f"Using encoder path from args: {encoder_path}")


    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    else:
        device = "cpu"
        print("GPU not detected, using CPU")

    experiment_name = cfg.get("experiment", "experiment")

    if args.checkpoint_mount is not None:
        output_dir = Path(args.checkpoint_mount)
    elif args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        split_mode = cfg["splits"].get("mode", "fold")
        label = split_mode if split_mode == "dev" else f"fold_{cfg['splits'].get('fold', 0)}"
        output_dir = Path("outputs") / label
    output_dir.mkdir(parents=True, exist_ok=True)

    best_output_dir = Path("outputs")
    best_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Latest checkpoint will be saved to: {output_dir.resolve()}")
    print(f"Best model will be saved to: {best_output_dir.resolve()}")

    
    paths = PanopTILsPaths(root=cfg["data"]["root"], subset=cfg["data"]["subset"])
    tmp_ds = PanopTILsDataset(paths=paths, file_list=None, transforms=None, cache_dataset=False, include_tissue_label=False)
    all_files = tmp_ds.files

    split_mode = cfg["splits"].get("mode", "fold")
    print(f"\nDataset: {cfg['data']['subset']}")
    print(f"Total files: {len(all_files)}")
    if split_mode == "dev":
        num_val_h = cfg["splits"].get("num_val_hospitals", 10)
        train_files, val_files = load_dev_split(all_files, root=cfg["data"]["root"], num_val_hospitals=num_val_h)
        print(f"Dev mode: Train={len(train_files)}, Val={len(val_files)}\n")
    else:
        fold_id = cfg["splits"]["fold"]
        train_files, val_files = load_splits(all_files, root=cfg["data"]["root"], fold=fold_id)
        print(f"Fold {fold_id}/5: Train={len(train_files)}, Val={len(val_files)}\n")

    train_transforms = create_train_transforms(image_size=256)
    val_transforms = create_val_transforms(image_size=256)

    dm = PanopTILsDataModule(cfg=DataConfig(**cfg["data"]),
                             train_files=train_files,
                             val_files=val_files,
                             train_transforms=train_transforms,
                             val_transforms=val_transforms,)
    dm.setup()

    data_cfg = cfg["data"]
    num_nuclei_classes = data_cfg["num_nuclei_classes"]
    num_tissue_classes = data_cfg["num_tissue_classes"]
    unlabeled_class = data_cfg.get("nuclei_unlabeled_class", None)
    tissue_ignore_classes = data_cfg.get("tissue_ignore_classes", [0])

    tissue_fusion = cfg["model"].get("tissue_fusion", "none") 
    use_tissue_branch = cfg["model"].get("use_tissue_branch", False) or tissue_fusion != "none" 
    use_compat = cfg["model"].get("use_compatibility_constraint", False)
    compat_map = NUCLEI_TISSUE_COMPATIBILITY

    if use_tissue_branch:
        fusion_warmup = cfg["model"].get("fusion_warmup_epochs", 0)
        freeze_tissue = cfg["model"].get("freeze_tissue_after_fusion_warmup", True)
        tissue_encoder_type = cfg["model"].get("tissue_encoder_type", "cnn")
        fusion_embed_dim = cfg["model"].get("fusion_embed_dim", 64)
        fusion_reduction = cfg["model"].get("fusion_reduction", 4)
        tissue_encoder_kwargs = cfg["model"].get("tissue_encoder_kwargs", None)
        model = CellViTWithTissue(
            tissue_fusion=tissue_fusion,
            use_compatibility_constraint=use_compat,
            nuclei_tissue_compatibility=compat_map,
            fusion_warmup_epochs=fusion_warmup,
            freeze_tissue_after_fusion_warmup=freeze_tissue,
            tissue_encoder_type=tissue_encoder_type,
            tissue_encoder_kwargs=tissue_encoder_kwargs,
            fusion_embed_dim=fusion_embed_dim,
            fusion_reduction=fusion_reduction,
            num_nuclei_classes=num_nuclei_classes,
            num_tissue_classes=num_tissue_classes,
            embed_dim=cfg["model"]["embed_dim"],
            input_channels=3,
            depth=cfg["model"]["depth"],
            num_heads=cfg["model"]["num_heads"],
            extract_layers=cfg["model"]["extract_layers"],
        )
        posthoc = cfg["model"].get("use_posthoc_constraint", False)
        print(f"Using CellViTWithTissue (fusion={tissue_fusion}, compat={use_compat}, fusion_warmup={fusion_warmup},"
              f" freeze_tissue={freeze_tissue}, posthoc_constraint={posthoc})")
    else:
        model = CellViT(
            num_nuclei_classes=num_nuclei_classes,
            num_tissue_classes=num_tissue_classes,
            embed_dim=cfg["model"]["embed_dim"],
            input_channels=3,
            depth=cfg["model"]["depth"],
            num_heads=cfg["model"]["num_heads"],
            extract_layers=cfg["model"]["extract_layers"],
        )

    ckpt = cfg["model"].get("encoder_pretrained", None)
    if ckpt is not None and os.path.exists(ckpt):
        load_vit_dino_pretrained(model, ckpt)
    else:
        print(f"Warning: Encoder checkpoint not found at {ckpt}. Training from scratch")

    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="max-autotune")
        except Exception as e:
            print(f"torch.compile not available, continuing without: {e}")


    nt_class_weights = dm.nt_class_weights(num_nuclei_classes)
    print(f"NT class weights: {[f'{w:.3f}' for w in nt_class_weights]}")

    loss_cfg = cfg["loss"]
    shared_loss_kwargs = dict(
        lambda_np_ft=loss_cfg.get("lambda_np_ft", 1.0),
        lambda_np_dice=loss_cfg.get("lambda_np_dice", 1.0),
        lambda_hv_mse=loss_cfg.get("lambda_hv_mse", 2.5),
        lambda_hv_msge=loss_cfg.get("lambda_hv_msge", 8.0),
        lambda_nt_ft=loss_cfg.get("lambda_nt_ft", 0.5),
        lambda_nt_dice=loss_cfg.get("lambda_nt_dice", 0.2),
        lambda_nt_bce=loss_cfg.get("lambda_nt_bce", 0.5),
        lambda_tc_ce=loss_cfg.get("lambda_tc_ce", 0.0),
        ft_alpha=loss_cfg.get("ft_alpha", 0.7),
        ft_beta=loss_cfg.get("ft_beta", 0.3),
        ft_gamma=loss_cfg.get("ft_gamma", 4.0 / 3.0),
        ft_eps=loss_cfg.get("ft_eps", 1e-6),
        unlabeled_class=unlabeled_class,
        nt_class_weights=nt_class_weights,
    )

    ts_class_weights = None
    if use_tissue_branch:
        ts_class_weights = dm.ts_class_weights(num_tissue_classes, ignore_classes=set(tissue_ignore_classes))
        print(f"Tissue class weights: {[f'{w:.3f}' for w in ts_class_weights]}")
        loss_fn = CellViTTissueLoss(
            lambda_ts_ft=loss_cfg.get("lambda_ts_ft", 1.0),
            lambda_ts_dice=loss_cfg.get("lambda_ts_dice", 0.5),
            lambda_ts_ce=loss_cfg.get("lambda_ts_ce", 1.0),
            tissue_ignore_classes=tissue_ignore_classes,
            ts_class_weights=ts_class_weights,
            tissue_label_smoothing=loss_cfg.get("tissue_label_smoothing", 0.0),
            tissue_dedup=loss_cfg.get("tissue_dedup", True),
            tissue_use_focal=loss_cfg.get("tissue_use_focal", False),
            tissue_focal_gamma=loss_cfg.get("tissue_focal_gamma", 2.0),
            **shared_loss_kwargs,
        )
    else:
        loss_fn = CellViTMultiTaskLoss(**shared_loss_kwargs)
    
    tissue_lr = cfg["train"].get("tissue_lr", cfg["train"]["lr"])
    tissue_wd = cfg["train"].get("tissue_wd", cfg["train"]["wd"])

    if use_tissue_branch and hasattr(model, "tissue_encoder"):
        tissue_params = set(id(p) for p in model.tissue_encoder.parameters())
        base_params = [p for p in model.parameters() if id(p) not in tissue_params]
        param_groups = [
            {"params": base_params, "lr": cfg["train"]["lr"]},
            {"params": list(model.tissue_encoder.parameters()),
             "lr": tissue_lr, "weight_decay": tissue_wd},
        ]
        print(f"Optimizer: lr={cfg['train']['lr']}, tissue_lr={tissue_lr}")
    else:
        param_groups = model.parameters()

    optim = torch.optim.AdamW(param_groups, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["wd"], 
                              betas=cfg["train"].get("betas", (0.85, 0.95)))

    scheduler_type = cfg["train"].get("scheduler_type", "exponential")
    tissue_scheduler_gamma = cfg["train"].get("tissue_scheduler_gamma", cfg["train"].get("scheduler_gamma", 0.85))
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg["train"]["epochs"], 
                                                               eta_min=cfg["train"].get("scheduler_eta_min", 1e-5))
        print(f"Using CosineAnnealingLR scheduler (eta_min={cfg['train'].get('scheduler_eta_min', 1e-5)})")
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=cfg["train"].get("scheduler_gamma", 0.85))
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
        num_nuclei_classes=num_nuclei_classes,
        num_tissue_classes=num_tissue_classes,
        excluded_nuclei_classes=dm.cfg.excluded_nuclei_classes,
        tissue_ignore_classes=tissue_ignore_classes,
        posthoc_compat_map=compat_map if cfg["model"].get("use_posthoc_constraint", False) else None,
        use_tissue_branch=use_tissue_branch,
        oracle_tissue_mode=cfg["model"].get("oracle_tissue_training", False),
    )

    run_id = args.run_id or uuid.uuid4().hex[:12]
    print(f"Run ID: {run_id}")

    start_epoch = 0
    best_pq = 0.0
    best_epoch = 0
    checkpoint_to_load = None
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint_to_load = args.checkpoint
    else:
        auto_ckpt = output_dir / "latest_checkpoint.pth"
        if auto_ckpt.exists():
            ckpt_data = torch.load(str(auto_ckpt), map_location="cpu", weights_only=False)
            saved_run_id = ckpt_data.get('run_id')
            if saved_run_id == run_id:
                checkpoint_to_load = str(auto_ckpt)
                print(f"Auto-resume: found checkpoint with matching run_id={run_id}")
            else:
                print(f"Checkpoint found but run_id mismatch. Starting fresh")
            del ckpt_data

    if checkpoint_to_load:
        strict = not args.no_strict
        start_epoch, best_pq, best_epoch = trainer.load_checkpoint(checkpoint_to_load, strict=strict)
        if not strict:
            start_epoch = 0
            best_pq = 0.0
            best_epoch = 0
            trainer.reset_early_stopping()
            print(f"Loaded weights (strict=False), training from epoch 0")
        else:
            print(f"Resuming from epoch {start_epoch}, best_pq={best_pq:.4f} (epoch {best_epoch})")

    wandb_id_file = output_dir / "wandb_run_id.txt"
    wandb_run_id = None
    if start_epoch > 0 and wandb_id_file.exists():
        wandb_run_id = wandb_id_file.read_text().strip()
        print(f"Resuming wandb run: {wandb_run_id}")

    wandb.init(
        project="cellvit-panoptils",
        id=wandb_run_id,
        resume="allow",
        config={
            "experiment": experiment_name,
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
            "split_mode": cfg["splits"].get("mode", "fold"),
            "fold": cfg["splits"].get("fold", "dev"),
            "model": cfg["model"],
            "loss_weights": cfg["loss"]
        },
        name=f"{experiment_name}_{cfg['splits'].get('fold', 'dev')}"
    )

    wandb_id_file.write_text(wandb.run.id)

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    print(f"\nStarting training for {cfg['train']['epochs']} epochs")
    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Val batches per epoch: {len(val_loader)}\n")

    val_metric_interval = cfg["train"].get("val_metric_interval", 5)
    print(f"Full validation metrics every {val_metric_interval} epochs")

    log_image_interval = cfg["train"].get("log_image_interval", 10)
    print(f"Prediction images logged to wandb every {log_image_interval} epochs")

    try:
        from azureml.core import Run as _AzureRun
        _ctx = _AzureRun.get_context()
        azure_run = None if _ctx.id.startswith("OfflineRun") else _ctx
    except Exception:
        azure_run = None

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        print(f"\nEpoch {epoch}/{cfg['train']['epochs']}")
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")

        train_metrics = trainer.train_epoch(train_loader, epoch=epoch)

        freeze_encoder_epochs = cfg["train"].get("freeze_encoder_epochs", 25)
        is_last_epoch = (epoch == cfg["train"]["epochs"] - 1)
        compute_full = (epoch % val_metric_interval == 0) or is_last_epoch
        val_metrics = trainer.val_epoch(val_loader, epoch=epoch, compute_full_metrics=compute_full)

        scheduler.step()

        metrics = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "train/loss": train_metrics['loss'],
        }

        if val_metrics:
            metrics["val/loss"] = val_metrics['loss']

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

        if epoch % log_image_interval == 0 and len(val_loader) > 0:
            metrics.update(trainer.log_prediction_images(val_loader))

        wandb.log(metrics)

        print(f"\nEpoch {epoch}:")
        print(f"\tTrain Loss: {train_metrics['loss']:.4f}")
        if val_metrics:
            print(f"\tVal Loss: {val_metrics['loss']:.4f}")
            if 'pq' in val_metrics:
                print(f"\tVal PQ: {val_metrics['pq']:.4f}")
                print(f"\tVal F1: {val_metrics['f1']:.4f}")

        pq_key = 'pq_class_avg' if 'pq_class_avg' in val_metrics else 'pq'
        if pq_key in val_metrics and val_metrics[pq_key] > best_pq:
            best_pq = val_metrics[pq_key]
            best_epoch = epoch
            checkpoint_path = best_output_dir / "best_model.pth"
            trainer.save_checkpoint(str(checkpoint_path), epoch, val_metrics, best_pq=best_pq,
                                    best_epoch=best_epoch, run_id=run_id, weights_only=True)
            print(f"  New best model with {pq_key}: {best_pq:.4f}")

        latest_path = output_dir / "latest_checkpoint.pth"
        trainer.save_checkpoint(str(latest_path), epoch, val_metrics, best_pq=best_pq,
                                best_epoch=best_epoch, run_id=run_id)

        if epoch >= freeze_encoder_epochs and trainer.check_early_stopping(val_metrics):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    print(f"\nTraining completed")
    print(f"Best PQ: {best_pq:.4f} at epoch {best_epoch}\n")

    wandb.finish()


if __name__ == "__main__":
    main()
