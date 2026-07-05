# Copyright © 2025 Olena Kosharova, FIIT STU.
# Part of the "Tissue-Context CellViT Extension" (bachelor's thesis, FIIT STU).
# Licensed under the Apache License 2.0 with the Commons Clause restriction.
# See the LICENSE file in the project root for full terms.

import argparse
import dataclasses
import os
import sys
import uuid
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.nn as nn
import yaml

from data.datamodule import DataConfig, PanopTILsDataModule
from data.constants import NUCLEI_TISSUE_COMPATIBILITY
from data.splits import load_splits
from data.transforms import create_train_transforms, create_val_transforms
from datasets.panoptils import PanopTILsDataset, PanopTILsPaths
from model.cellvit_panoptils import CellViTWithTissue
from model.tissue_smp import SMPSegEncoder
from training.losses import CellViTTissueLoss, DiceLoss, FocalTverskyLoss
from training.trainer import Trainer
from src.train import load_vit_dino_pretrained


def _setup_datamodule(data_cfg: dict, train_files, val_files, overrides: dict = None) -> PanopTILsDataModule:
    cfg = {**data_cfg, **(overrides or {})}
    train_tx = create_train_transforms(image_size=256)
    val_tx = create_val_transforms(image_size=256)
    dc_fields = {f.name for f in dataclasses.fields(DataConfig)}
    dc_kwargs = {k: v for k, v in cfg.items() if k in dc_fields}
    dm = PanopTILsDataModule(
        cfg=DataConfig(**dc_kwargs),
        train_files=train_files,
        val_files=val_files,
        train_transforms=train_tx,
        val_transforms=val_tx,
    )
    dm.setup()
    return dm


class _TissueOnlyLoss:
    def __init__(self, ignore_classes, class_weights, lambda_dice, lambda_ft, lambda_ce,
                 ft_alpha, ft_beta, ft_gamma, ft_eps, label_smoothing, device):
        self.lambda_dice = lambda_dice
        self.lambda_ft = lambda_ft
        self.lambda_ce = lambda_ce
        self.dice = DiceLoss(ignore_classes=ignore_classes) if lambda_dice > 0 else None
        self.ft = FocalTverskyLoss( alpha=ft_alpha, beta=ft_beta, gamma=ft_gamma, smooth=ft_eps,
                                    ignore_classes=ignore_classes) if lambda_ft > 0 else None
        weight = torch.tensor(class_weights, dtype=torch.float32).to(device) if class_weights else None
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_classes[0] if ignore_classes else -100,
                                      label_smoothing=label_smoothing)

    def __call__(self, logits, targets):
        loss = torch.tensor(0.0, device=logits.device)
        if self.lambda_ce > 0:
            loss = loss + self.lambda_ce * self.ce(logits, targets.long())
        if self.dice is not None:
            loss = loss + self.lambda_dice * self.dice(logits, targets)
        if self.ft is not None:
            loss = loss + self.lambda_ft * self.ft(logits, targets)
        return loss


def run_phase1(cfg: dict, output_dir: Path, train_files, val_files, device: str, args) -> Path:
    p1_cfg = cfg["phase1"]
    data_cfg = {**cfg["data"], **p1_cfg.get("data_overrides", {})}
    train_cfg = p1_cfg["train"]
    loss_cfg = p1_cfg["loss"]
    epochs = p1_cfg["epochs"]

    latest_path = output_dir / "phase1_latest.pth"
    best_path = output_dir / "phase1_best.pth"

    start_epoch = 0
    best_dice = 0.0
    best_epoch = 0

    tissue_kwargs = cfg["model"].get("tissue_encoder_kwargs", {})
    model = SMPSegEncoder(num_classes=data_cfg["num_tissue_classes"], **tissue_kwargs).to(device)

    dm = _setup_datamodule(data_cfg, train_files, val_files)

    ignore_classes = data_cfg.get("tissue_ignore_classes", [0])
    ts_weights = None
    if loss_cfg.get("use_inv_freq_weights", False):
        ts_weights = dm.ts_class_weights(data_cfg["num_tissue_classes"], ignore_classes=set(ignore_classes))

    loss_fn = _TissueOnlyLoss(
        ignore_classes=ignore_classes,
        class_weights=ts_weights,
        lambda_dice=loss_cfg.get("lambda_ts_dice", 0.5),
        lambda_ft=loss_cfg.get("lambda_ts_ft", 0.0),
        lambda_ce=loss_cfg.get("lambda_ts_ce", 0.5),
        ft_alpha=loss_cfg.get("ft_alpha", 0.4),
        ft_beta=loss_cfg.get("ft_beta", 0.6),
        ft_gamma=loss_cfg.get("ft_gamma", 1.3333),
        ft_eps=loss_cfg.get("ft_eps", 1e-6),
        label_smoothing=loss_cfg.get("tissue_label_smoothing", 0.0),
        device=device,
    )

    freeze_epochs = train_cfg.get("encoder_freeze_epochs", 0)
    if freeze_epochs > 0 and hasattr(model, "freeze_encoder"):
        model.freeze_encoder()

    optim = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["wd"], 
                              betas=train_cfg.get("betas", (0.9, 0.999)))

    sched_type = train_cfg.get("scheduler_type", "multistep")
    if sched_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs, 
                                                               eta_min=train_cfg.get("scheduler_eta_min", 1e-6))
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=train_cfg.get("scheduler_milestones", [40, 70]),
                                                         gamma=train_cfg.get("scheduler_gamma", 0.5))

    scaler = torch.amp.GradScaler(enabled=train_cfg.get("use_mixed_precision", True))
    use_amp = train_cfg.get("use_mixed_precision", True) and device == "cuda"

    if latest_path.exists():
        ckpt = torch.load(str(latest_path), map_location="cpu", weights_only=False)
        if ckpt.get("run_id") == args.run_id:
            model.load_state_dict(ckpt["model"])
            optim.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_dice = ckpt.get("best_dice", 0.0)
            best_epoch = ckpt.get("best_epoch", 0)
            print(f"Resumed from epoch {start_epoch}, best_dice={best_dice:.4f}")

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    patience = train_cfg.get("early_stopping_patience", 30)
    no_improve = 0

    print(f"Training for {epochs} epochs")
    for epoch in range(start_epoch, epochs):
        if freeze_epochs > 0 and epoch == freeze_epochs and hasattr(model, "unfreeze_encoder"):
            model.unfreeze_encoder()
            print(f"Unfreezing encoder")

        model.train()
        train_loss = 0.0
        for _, targets, _ in train_loader:
            imgs = targets["tissue_context"].to(device)
            tissue_gt = targets["tissue_mask_context"].to(device)
            optim.zero_grad()
            with torch.amp.autocast(enabled=use_amp):
                logits, _ = model(imgs)
                loss = loss_fn(logits, tissue_gt)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("max_grad_norm", 1.0))
            scaler.step(optim)
            scaler.update()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        dice_sum, n_val = 0.0, 0
        with torch.no_grad():
            for _, targets, _ in val_loader:
                imgs = targets["tissue_context"].to(device)
                tissue_gt = targets["tissue_mask_context"].to(device)
                with torch.amp.autocast(enabled=use_amp):
                    logits, _ = model(imgs)
                    loss = loss_fn(logits, tissue_gt)
                val_loss += loss.item()
                pred = logits.argmax(1)
                for c in range(1, data_cfg["num_tissue_classes"]):
                    if c in ignore_classes:
                        continue
                    tp = ((pred == c) & (tissue_gt == c)).sum().float()
                    fp = ((pred == c) & (tissue_gt != c)).sum().float()
                    fn = ((pred != c) & (tissue_gt == c)).sum().float()
                    denom = 2 * tp + fp + fn
                    if denom > 0:
                        dice_sum += (2 * tp / denom).item()
                        n_val += 1
        val_loss /= len(val_loader)
        tissue_dice = dice_sum / n_val if n_val > 0 else 0.0
        print(f"Epoch {epoch}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  tissue_dice={tissue_dice:.4f}")

        if args.use_wandb:
            import wandb
            wandb.log({"phase": 1, "epoch": epoch,
                       "p1/train_loss": train_loss,
                       "p1/val_loss": val_loss,
                       "p1/tissue_dice": tissue_dice})

        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optim.state_dict(), 
                    "scheduler": scheduler.state_dict(), "best_dice": best_dice, "best_epoch": best_epoch,
                    "run_id": args.run_id}, str(latest_path))

        if tissue_dice > best_dice:
            best_dice = tissue_dice
            best_epoch = epoch
            no_improve = 0
            torch.save({"epoch": epoch, "model": model.state_dict(), "tissue_dice": best_dice}, str(best_path))
            print(f"New best tissue_dice={best_dice:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Phase 1 is done. Best tissue_dice={best_dice:.4f} at epoch {best_epoch}")
    return best_path


def run_phase2(cfg: dict, phase1_ckpt: Path, output_dir: Path, train_files, val_files, device: str, args) -> None:
    p2_cfg = cfg["phase2"]
    data_cfg = cfg["data"]
    train_cfg = p2_cfg["train"]
    loss_cfg = p2_cfg["loss"]
    mcfg = cfg["model"]
    compat_map = NUCLEI_TISSUE_COMPATIBILITY

    dm = _setup_datamodule(data_cfg, train_files, val_files)

    model = CellViTWithTissue(
        tissue_fusion=mcfg.get("tissue_fusion", "none"),
        use_compatibility_constraint=mcfg.get("use_compatibility_constraint", True),
        nuclei_tissue_compatibility=compat_map,
        fusion_warmup_epochs=mcfg.get("fusion_warmup_epochs", 0),
        freeze_tissue_after_fusion_warmup=mcfg.get("freeze_tissue_after_fusion_warmup", True),
        tissue_encoder_type="smp",
        tissue_encoder_kwargs=mcfg.get("tissue_encoder_kwargs", {}),
        num_nuclei_classes=data_cfg["num_nuclei_classes"],
        num_tissue_classes=data_cfg["num_tissue_classes"],
        embed_dim=mcfg["embed_dim"],
        input_channels=3,
        depth=mcfg["depth"],
        num_heads=mcfg["num_heads"],
        extract_layers=mcfg["extract_layers"],
    )

    dino_path = mcfg.get("encoder_pretrained")
    if dino_path and os.path.exists(dino_path):
        load_vit_dino_pretrained(model, dino_path)
    else:
        print(f"Warning: DINO checkpoint not found at {dino_path}")

    p1_state = torch.load(str(phase1_ckpt), map_location="cpu", weights_only=False)
    missing, unexpected = model.tissue_encoder.load_state_dict(p1_state["model"], strict=False)

    if missing:
        print(f"Warning: {len(missing)} missing keys loading tissue weights")
    for p in model.tissue_encoder.parameters():
        p.requires_grad = False

    model.tissue_encoder.eval()
    model._keep_tissue_eval = True
    model = model.to(device)

    nt_class_weights = dm.nt_class_weights(data_cfg["num_nuclei_classes"])
    ts_class_weights = dm.ts_class_weights(data_cfg["num_tissue_classes"], 
                                           ignore_classes=set(data_cfg.get("tissue_ignore_classes", [0])))

    loss_fn = CellViTTissueLoss(
        lambda_np_ft=loss_cfg.get("lambda_np_ft", 1.0),
        lambda_np_dice=loss_cfg.get("lambda_np_dice", 1.0),
        lambda_hv_mse=loss_cfg.get("lambda_hv_mse", 2.5),
        lambda_hv_msge=loss_cfg.get("lambda_hv_msge", 8.0),
        lambda_nt_ft=loss_cfg.get("lambda_nt_ft", 1.0),
        lambda_nt_dice=loss_cfg.get("lambda_nt_dice", 0.5),
        lambda_nt_bce=loss_cfg.get("lambda_nt_bce", 1.0),
        lambda_ts_ft=loss_cfg.get("lambda_ts_ft", 0.0),
        lambda_ts_dice=loss_cfg.get("lambda_ts_dice", 0.0),
        lambda_ts_ce=loss_cfg.get("lambda_ts_ce", 0.0),
        lambda_tc_ce=loss_cfg.get("lambda_tc_ce", 0.0),
        tissue_ignore_classes=data_cfg.get("tissue_ignore_classes", [0]),
        ts_class_weights=ts_class_weights,
        tissue_label_smoothing=loss_cfg.get("tissue_label_smoothing", 0.0),
        tissue_dedup=loss_cfg.get("tissue_dedup", True),
        ft_alpha=loss_cfg.get("ft_alpha", 0.4),
        ft_beta=loss_cfg.get("ft_beta", 0.6),
        ft_gamma=loss_cfg.get("ft_gamma", 1.3333),
        ft_eps=loss_cfg.get("ft_eps", 1e-6),
        unlabeled_class=data_cfg.get("nuclei_unlabeled_class"),
        nt_class_weights=nt_class_weights,
    )

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], 
                              lr=train_cfg["lr"], weight_decay=train_cfg["wd"],
                              betas=train_cfg.get("betas", (0.9, 0.999)))

    sched_type = train_cfg.get("scheduler_type", "cosine")
    if sched_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=p2_cfg["epochs"],
                                                               eta_min=train_cfg.get("scheduler_eta_min", 1e-5))
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, 
                                                         milestones=train_cfg.get("scheduler_milestones", [60, 80]),
                                                         gamma=train_cfg.get("scheduler_gamma", 0.5))

    trainer = Trainer(
        model=model, loss_fn=loss_fn, optimizer=optim, scheduler=scheduler,
        device=device,
        use_mixed_precision=train_cfg.get("use_mixed_precision", True),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        freeze_encoder_epochs=train_cfg.get("freeze_encoder_epochs", 0),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        early_stopping_patience=train_cfg.get("early_stopping_patience"),
        num_nuclei_classes=data_cfg["num_nuclei_classes"],
        num_tissue_classes=data_cfg["num_tissue_classes"],
        excluded_nuclei_classes=dm.cfg.excluded_nuclei_classes,
        tissue_ignore_classes=data_cfg.get("tissue_ignore_classes", [0]),
        posthoc_compat_map=compat_map if mcfg.get("use_posthoc_constraint", False) else None,
        oracle_tissue_mode=mcfg.get("oracle_tissue_training", False),
    )

    latest_path = output_dir / "phase2_latest.pth"
    best_path = Path("./outputs/phase2_best.pth")
    best_path.parent.mkdir(parents=True, exist_ok=True)

    start_epoch, best_pq, best_epoch = 0, 0.0, 0
    if latest_path.exists():
        ckpt = torch.load(str(latest_path), map_location="cpu", weights_only=False)
        if ckpt.get("run_id") == args.run_id:
            start_epoch, best_pq, best_epoch = trainer.load_checkpoint(str(latest_path), strict=False)
            for p in model.tissue_encoder.parameters():
                p.requires_grad = False
            model._keep_tissue_eval = True
            model.tissue_encoder.eval()
            print(f"Resumed from epoch {start_epoch}, best_pq={best_pq:.4f}")

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    val_metric_interval = train_cfg.get("val_metric_interval", 5)
    log_image_interval = train_cfg.get("log_image_interval", 10)

    for epoch in range(start_epoch, p2_cfg["epochs"]):
        print(f"\nEpoch {epoch}/{p2_cfg['epochs']} lr={scheduler.get_last_lr()[0]:.6f}")
        train_metrics = trainer.train_epoch(train_loader, epoch=epoch)

        compute_full = (epoch % val_metric_interval == 0) or (epoch == p2_cfg["epochs"] - 1)
        val_metrics = trainer.val_epoch(val_loader, epoch=epoch, compute_full_metrics=compute_full)
        scheduler.step()

        if args.use_wandb:
            import wandb
            log = {"phase": 2, "epoch": epoch, "lr": scheduler.get_last_lr()[0]}
            for k, v in train_metrics.items():
                log[f"p2/train/{k}"] = v
            for k, v in val_metrics.items():
                log[f"p2/val/{k}"] = v
            if epoch % log_image_interval == 0:
                log.update(trainer.log_prediction_images(val_loader))
            wandb.log(log)

        pq_key = "pq_class_avg" if "pq_class_avg" in val_metrics else "pq"
        if pq_key in val_metrics and val_metrics[pq_key] > best_pq:
            best_pq = val_metrics[pq_key]
            best_epoch = epoch
            trainer.save_checkpoint(str(best_path), epoch, val_metrics, best_pq=best_pq, 
                                    best_epoch=best_epoch, run_id=args.run_id, weights_only=True)
            print(f"New best {pq_key}={best_pq:.4f}")

        trainer.save_checkpoint(str(latest_path), epoch, val_metrics, best_pq=best_pq, 
                                best_epoch=best_epoch, run_id=args.run_id)

    print(f"\nPhase 2 is done. Best PQ={best_pq:.4f} at epoch {best_epoch}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cellvit_smp_constraint.yaml")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--dataset-subdir", default=None)
    parser.add_argument("--encoder-path", default=None)
    parser.add_argument("--encoder-filename", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--phase1-checkpoint", default=None,
                        help="Skip phase 1 and load tissue weights from this path")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.dataset_path:
        root = args.dataset_path
        if args.dataset_subdir:
            root = os.path.join(root, args.dataset_subdir)
        cfg["data"]["root"] = root

    if args.encoder_path:
        ep = args.encoder_path
        if args.encoder_filename:
            ep = os.path.join(ep, args.encoder_filename)
        cfg["model"]["encoder_pretrained"] = ep

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    fold_id = cfg["splits"]["fold"]
    output_dir = (Path(args.output_dir) if args.output_dir else Path("outputs/two_stage") / f"fold_{fold_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir.resolve()}")

    paths = PanopTILsPaths(root=cfg["data"]["root"], subset=cfg["data"]["subset"])
    tmp_ds = PanopTILsDataset(paths=paths, file_list=None, transforms=None,
                              cache_dataset=False, include_tissue_label=False)
    train_files, val_files = load_splits(tmp_ds.files, root=cfg["data"]["root"], fold=fold_id)
    print(f"Fold {fold_id}: train={len(train_files)}, val={len(val_files)}")

    args.run_id = args.run_id or uuid.uuid4().hex[:12]
    print(f"Run ID: {args.run_id}")

    args.use_wandb = not args.no_wandb
    if args.use_wandb:
        import wandb
        wandb.init(
            project="cellvit-panoptils-two-stage",
            id=args.run_id,
            resume="allow",
            config=cfg,
            name=f"two_stage_fold{fold_id}",
        )

    if args.phase1_checkpoint:
        phase1_best = Path(args.phase1_checkpoint)
        print(f"Skipping phase 1\nUsing tissue checkpoint: {phase1_best}")
    else:
        phase1_best = run_phase1(cfg, output_dir, train_files, val_files, device, args)

    run_phase2(cfg, phase1_best, output_dir, train_files, val_files, device, args)

    if args.use_wandb:
        import wandb
        wandb.finish()

    print("\nThe whole training complete")


if __name__ == "__main__":
    main()
