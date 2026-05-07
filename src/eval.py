import argparse
import os
import sys
from pathlib import Path
import torch
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.datamodule import DataConfig, PanopTILsDataModule
from data.constants import NUCLEI_TISSUE_COMPATIBILITY
from data.splits import load_splits, load_test_split
from data.transforms import create_train_transforms, create_val_transforms
from datasets.panoptils import PanopTILsDataset, PanopTILsPaths
from model.cellvit import CellViT
from model.cellvit_panoptils import CellViTWithTissue
from training.losses import CellViTMultiTaskLoss, CellViTTissueLoss
from training.trainer import Trainer


def build_model(cfg, num_nuclei_classes, num_tissue_classes, compat_map):
    mcfg = cfg["model"]
    tissue_fusion = mcfg.get("tissue_fusion", "none")
    use_tissue_branch = mcfg.get("use_tissue_branch", False) or tissue_fusion != "none"

    if use_tissue_branch:
        return CellViTWithTissue(
            tissue_fusion=tissue_fusion,
            use_compatibility_constraint=mcfg.get("use_compatibility_constraint", False),
            nuclei_tissue_compatibility=compat_map,
            fusion_warmup_epochs=mcfg.get("fusion_warmup_epochs", 0),
            freeze_tissue_after_fusion_warmup=mcfg.get("freeze_tissue_after_fusion_warmup", True),
            tissue_encoder_type=mcfg.get("tissue_encoder_type", "cnn"),
            tissue_encoder_kwargs=mcfg.get("tissue_encoder_kwargs", {}),
            fusion_embed_dim=mcfg.get("fusion_embed_dim", 64),
            fusion_reduction=mcfg.get("fusion_reduction", 4),
            num_nuclei_classes=num_nuclei_classes,
            num_tissue_classes=num_tissue_classes,
            embed_dim=mcfg["embed_dim"],
            input_channels=3,
            depth=mcfg["depth"],
            num_heads=mcfg["num_heads"],
            extract_layers=mcfg["extract_layers"],
        ), True
    else:
        return CellViT(
            num_nuclei_classes=num_nuclei_classes,
            num_tissue_classes=num_tissue_classes,
            embed_dim=mcfg["embed_dim"],
            input_channels=3,
            depth=mcfg["depth"],
            num_heads=mcfg["num_heads"],
            extract_layers=mcfg["extract_layers"],
        ), False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="Override dataset root path")
    parser.add_argument("--dataset-subdir", type=str, default=None,
                        help="Subdirectory under --dataset-path")
    parser.add_argument("--use-posthoc", action="store_true",
                        help="Apply post-hoc tissue compatibility constraint at inference")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch_size for evaluation")
    parser.add_argument("--test-set", action="store_true",
                        help="Evaluate on held-out test.csv instead of the fold val set")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.dataset_path is not None:
        root = args.dataset_path
        if args.dataset_subdir:
            root = os.path.join(root, args.dataset_subdir)
        cfg["data"]["root"] = root

    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size

    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    paths = PanopTILsPaths(root=cfg["data"]["root"], subset=cfg["data"]["subset"])
    tmp_ds = PanopTILsDataset(paths=paths, file_list=None, transforms=None,
                              cache_dataset=False, include_tissue_label=False)

    if args.test_set:
        eval_files = load_test_split(tmp_ds.files, root=cfg["data"]["root"])
        train_files = []
        val_files = eval_files
        print(f"Test set: {len(eval_files)} patches")
    else:
        fold_id = cfg["splits"]["fold"]
        train_files, val_files = load_splits(tmp_ds.files, root=cfg["data"]["root"], fold=fold_id)
        print(f"Fold {fold_id}: val={len(val_files)}")

    cfg["data"]["cache_dataset"] = False

    dm = PanopTILsDataModule(
        cfg=DataConfig(**cfg["data"]),
        train_files=train_files,
        val_files=val_files,
        train_transforms=create_train_transforms(image_size=256),
        val_transforms=create_val_transforms(image_size=256),
    )
    dm.setup()

    data_cfg = cfg["data"]
    num_nuclei_classes = data_cfg["num_nuclei_classes"]
    num_tissue_classes = data_cfg["num_tissue_classes"]
    compat_map = NUCLEI_TISSUE_COMPATIBILITY

    model, use_tissue_branch = build_model(cfg, num_nuclei_classes, num_tissue_classes, compat_map)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    state = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Missing keys: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")

    print(f"Checkpoint epoch: {ckpt.get('epoch', '?')}, best_pq: {ckpt.get('best_pq', '?')}")

    nt_class_weights = (dm.nt_class_weights(num_nuclei_classes) if dm.train_ds is not None else [1.0] * num_nuclei_classes)

    loss_cfg = cfg.get("loss", {})
    shared = dict(
        lambda_np_ft=loss_cfg.get("lambda_np_ft", 1.0),
        lambda_np_dice=loss_cfg.get("lambda_np_dice", 1.0),
        lambda_hv_mse=loss_cfg.get("lambda_hv_mse", 2.5),
        lambda_hv_msge=loss_cfg.get("lambda_hv_msge", 8.0),
        lambda_nt_ft=loss_cfg.get("lambda_nt_ft", 1.0),
        lambda_nt_dice=loss_cfg.get("lambda_nt_dice", 0.5),
        lambda_nt_bce=loss_cfg.get("lambda_nt_bce", 1.0),
        lambda_tc_ce=loss_cfg.get("lambda_tc_ce", 0.0),
        ft_alpha=loss_cfg.get("ft_alpha", 0.4),
        ft_beta=loss_cfg.get("ft_beta", 0.6),
        ft_gamma=loss_cfg.get("ft_gamma", 1.3333),
        ft_eps=loss_cfg.get("ft_eps", 1e-6),
        unlabeled_class=data_cfg.get("nuclei_unlabeled_class"),
        nt_class_weights=nt_class_weights,
    )

    if use_tissue_branch:
        if dm.train_ds is not None:
            ts_class_weights = dm.ts_class_weights(num_tissue_classes,
                                ignore_classes=set(data_cfg.get("tissue_ignore_classes", [0])))
        else:
            ts_class_weights = [1.0] * num_tissue_classes

        loss_fn = CellViTTissueLoss(
            lambda_ts_ft=loss_cfg.get("lambda_ts_ft", 1.0),
            lambda_ts_dice=loss_cfg.get("lambda_ts_dice", 0.5),
            lambda_ts_ce=loss_cfg.get("lambda_ts_ce", 1.0),
            tissue_ignore_classes=data_cfg.get("tissue_ignore_classes", [0]),
            ts_class_weights=ts_class_weights,
            tissue_label_smoothing=loss_cfg.get("tissue_label_smoothing", 0.0),
            tissue_dedup=loss_cfg.get("tissue_dedup", True),
            tissue_use_focal=loss_cfg.get("tissue_use_focal", False),
            tissue_focal_gamma=loss_cfg.get("tissue_focal_gamma", 2.0),
            **shared,
        )
    else:
        loss_fn = CellViTMultiTaskLoss(**shared)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)  

    oracle_tissue_mode = cfg["model"].get("oracle_tissue_training", False)
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optim,
        device=device,
        use_mixed_precision=(device == "cuda"),
        num_nuclei_classes=num_nuclei_classes,
        num_tissue_classes=num_tissue_classes,
        excluded_nuclei_classes=dm.cfg.excluded_nuclei_classes,
        tissue_ignore_classes=data_cfg.get("tissue_ignore_classes", [0]),
        posthoc_compat_map=compat_map if args.use_posthoc else None,
        use_tissue_branch=use_tissue_branch,
        oracle_tissue_mode=oracle_tissue_mode,
    )

    val_loader = dm.val_dataloader()
    print(f"\nRunning validation ({'with' if args.use_posthoc else 'without'} post-hoc constraint)")
    metrics = trainer.val_epoch(val_loader, epoch=0, compute_full_metrics=True)

    print("\n\nResults\n")
    for k in sorted(metrics.keys()):
        v = metrics[k]
        if isinstance(v, (int, float)):
            print(f"\t{k:30s}\t{v:.4f}")

    print("\nDone")


if __name__ == "__main__":
    main()
