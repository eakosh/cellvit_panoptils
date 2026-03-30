import os
import torch
from tqdm import tqdm
import numpy as np
from typing import Dict

from model.utils.post_proc_cellvit import DetectionCellPostProcessor
from utils.metrics import MetricsAggregator, compute_tissue_dice


class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def step(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (score > self.best_score) if self.mode == "max" else (score < self.best_score)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        return False


class Trainer:
    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        scheduler=None,
        device="cuda",
        use_mixed_precision=True,
        gradient_accumulation_steps=1,
        freeze_encoder_epochs=25,
        max_grad_norm=1.0,
        early_stopping_patience=None,
        num_nuclei_classes=10,
        num_tissue_classes=9,
        unlabeled_class=None,
        background_class=None,
        ambiguous_classes=(),
        centroid_radius=12.0,
        tissue_ignore_classes=None,
    ):
        self.num_tissue_classes = num_tissue_classes
        self.tissue_ignore_classes = set(tissue_ignore_classes) if tissue_ignore_classes else {0}
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.use_mixed_precision = use_mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.freeze_encoder_epochs = freeze_encoder_epochs
        self.max_grad_norm = max_grad_norm
        self.num_nuclei_classes = num_nuclei_classes

        if self.use_mixed_precision:
            self.scaler = torch.amp.GradScaler()
        else:
            self.scaler = None

        self.metrics_aggregator = MetricsAggregator(
            num_classes=num_nuclei_classes,
            unlabeled_class=unlabeled_class,
            background_class=background_class,
            ambiguous_classes=tuple(ambiguous_classes),
            centroid_radius=centroid_radius,
        )

        self.cell_post_processor = DetectionCellPostProcessor(
            nr_types=num_nuclei_classes,
            magnification=40,
            gt=False,
        )

        if early_stopping_patience is not None:
            self.early_stopping = EarlyStopping(patience=early_stopping_patience, mode="min")
        else:
            self.early_stopping = None

        self.current_epoch = 0

    def freeze_encoder(self):
        if hasattr(self.model, 'encoder'):
            for layer_name, param in self.model.encoder.named_parameters():
                if layer_name.split(".")[0] != "head":
                    param.requires_grad = False
        else:
            print("Warning: Model doesn't have encoder attribute. Skipping freeze")

    def unfreeze_encoder(self):
        if hasattr(self.model, 'encoder'):
            for param in self.model.encoder.parameters():
                param.requires_grad = True
        else:
            print("Warning: Model doesn't have encoder attribute. Skipping unfreeze")

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

        if epoch < self.freeze_encoder_epochs:
            self.freeze_encoder()
        else:
            if epoch == self.freeze_encoder_epochs:
                print(f"\nEpoch {epoch}: Unfreezing encoder\n")
            self.unfreeze_encoder()

    def train_epoch(self, loader, epoch: int):
        self.set_epoch(epoch)
        self.model.train()

        total_loss = 0.0
        loss_components = {}
        batch_count = 0

        num_batches = len(loader)
        pbar = tqdm(loader, desc=f"Train Epoch {epoch}")

        for batch_idx, (images, targets, names) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            targets = {k: v.to(self.device, non_blocking=True) for k, v in targets.items()}

            # forward
            if self.use_mixed_precision:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss, loss_dict = self.loss_fn(outputs, targets)
                    loss = loss / self.gradient_accumulation_steps
            else:
                outputs = self.model(images)
                loss, loss_dict = self.loss_fn(outputs, targets)
                loss = loss / self.gradient_accumulation_steps

            # backward
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # gradient accumulation
            if ((batch_idx + 1) % self.gradient_accumulation_steps == 0
                    or (batch_idx + 1) == num_batches):
                if self.use_mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)

            # loss
            batch_loss = loss.item() * self.gradient_accumulation_steps
            total_loss += batch_loss
            batch_count += 1

            for key, value in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value

            pbar.set_postfix({'loss': f'{batch_loss:.4f}'})

        avg_loss = total_loss / max(1, batch_count)
        avg_loss_components = {k: v / max(1, batch_count) for k, v in loss_components.items()}

        metrics = {
            'loss': avg_loss,
            **avg_loss_components
        }

        return metrics

    @torch.no_grad()
    def val_epoch(self, loader, epoch: int, compute_full_metrics: bool = True):
        self.model.eval()
        if compute_full_metrics:
            self.metrics_aggregator.reset()

        total_loss = 0.0
        loss_components = {}
        batch_count = 0
        
        tissue_acc = {}
        tissue_n = 0

        desc = f"Val Epoch {epoch}" + (" [full]" if compute_full_metrics else " [loss only]")
        pbar = tqdm(loader, desc=desc)

        for images, targets, names in pbar:
            images = images.to(self.device, non_blocking=True)
            targets = {k: v.to(self.device, non_blocking=True) for k, v in targets.items()}

            # forward
            if self.use_mixed_precision:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss, loss_dict = self.loss_fn(outputs, targets)
            else:
                outputs = self.model(images)
                loss, loss_dict = self.loss_fn(outputs, targets)

            batch_loss = loss.item()
            total_loss += batch_loss
            batch_count += 1

            for key, value in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value

            if compute_full_metrics:
                pred_instances = self._extract_instances_hv(outputs)
                gt_instances = targets['instance_map'].cpu().numpy()

                pred_type = torch.argmax(outputs['nuclei_type_map'], dim=1).cpu().numpy()
                gt_type = targets['nuclei_type_map'].cpu().numpy()
                pred_binary = (torch.softmax(outputs['nuclei_binary_map'], dim=1)[:, 1] > 0.5).cpu().numpy().astype(np.uint8)
                gt_binary = targets['nuclei_binary_map'].cpu().numpy().astype(np.uint8)

                for i in range(pred_instances.shape[0]):
                    self.metrics_aggregator.update(
                        pred_instances[i],
                        gt_instances[i],
                        pred_binary=pred_binary[i],
                        gt_binary=gt_binary[i],
                        pred_type_map=pred_type[i],
                        gt_type_map=gt_type[i],
                    )

                # tissue dice
                if "tissue_segmentation_map" in outputs:
                    pred_t = torch.argmax(outputs["tissue_segmentation_map"], dim=1).cpu().numpy()
                    gt_t = targets["tissue_mask"].cpu().numpy()
                    for i in range(pred_t.shape[0]):
                        td = compute_tissue_dice(
                            pred_t[i], gt_t[i],
                            num_classes=self.num_tissue_classes,
                            ignore_classes=self.tissue_ignore_classes,
                        )
                        for k, v in td.items():
                            tissue_acc[k] = tissue_acc.get(k, 0.0) + v
                        tissue_n += 1

            pbar.set_postfix({'loss': f'{batch_loss:.4f}'})

        avg_loss = total_loss / max(1, batch_count)
        avg_loss_components = {k: v / max(1, batch_count) for k, v in loss_components.items()}

        metrics = {
            'loss': avg_loss,
            **avg_loss_components,
        }

        if compute_full_metrics:
            val_metrics = self.metrics_aggregator.compute()
            metrics.update(val_metrics)
            print(f"\nValidation Metrics - {self.metrics_aggregator}\n")

            if tissue_n > 0 and "mean" in tissue_acc:
                metrics["tissue_dice"] = tissue_acc["mean"] / tissue_n
                for c in range(1, self.num_tissue_classes):
                    if c in tissue_acc:
                        metrics[f"tissue_dice_class_{c}"] = tissue_acc[c] / tissue_n

        return metrics

    def _extract_instances_hv(self, outputs: Dict[str, torch.Tensor]) -> np.ndarray:
        binary_pred = torch.softmax(outputs['nuclei_binary_map'], dim=1)
        type_pred = torch.argmax(outputs['nuclei_type_map'], dim=-1) if outputs['nuclei_type_map'].dim() == 3 \
            else torch.argmax(outputs['nuclei_type_map'], dim=1)
        hv_pred = outputs['hv_map']

        batch_size = binary_pred.shape[0]
        instance_maps = []

        for i in range(batch_size):
            # pred_map: [H, W, 4] = [type, binary_prob, hv_x, hv_y]
            pred_map = np.concatenate([
                type_pred[i].detach().cpu().numpy()[..., None],                    
                binary_pred[i, 1].detach().cpu().numpy()[..., None],               
                hv_pred[i, 0].detach().cpu().numpy()[..., None],                   
                hv_pred[i, 1].detach().cpu().numpy()[..., None],                   
            ], axis=-1)

            instance_map, _ = self.cell_post_processor.post_process_cell_segmentation(pred_map)
            instance_maps.append(instance_map)

        return np.stack(instance_maps, axis=0)

    @torch.no_grad()
    def log_prediction_images(self, loader, n_samples: int = 4) -> dict:
        import wandb
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        self.model.eval()

        images, targets, names = next(iter(loader))
        images = images.to(self.device, non_blocking=True)
        targets = {k: v.to(self.device, non_blocking=True) for k, v in targets.items()}

        n_samples = min(n_samples, images.shape[0])

        if self.use_mixed_precision:
            with torch.amp.autocast('cuda'):
                outputs = self.model(images)
        else:
            outputs = self.model(images)

        # binary
        pred_binary = torch.softmax(outputs['nuclei_binary_map'], dim=1)[:, 1].cpu().numpy()
        pred_type = torch.argmax(outputs['nuclei_type_map'], dim=1).cpu().numpy()
        pred_hv = outputs['hv_map'].cpu().numpy()  # [B, 2, H, W]

        gt_binary = targets['nuclei_binary_map'].cpu().numpy().astype(np.float32)
        gt_type = targets['nuclei_type_map'].cpu().numpy()
        gt_hv = targets['hv_map'].cpu().numpy()  # [B, 2, H, W]

        # tissue segmentation
        has_tissue = "tissue_segmentation_map" in outputs
        if has_tissue:
            pred_tissue = torch.argmax(outputs['tissue_segmentation_map'], dim=1).cpu().numpy()
            gt_tissue = targets['tissue_mask'].cpu().numpy()

        # denormalize images
        imgs_np = images.cpu().numpy()  # [B, C, H, W]
        imgs_np = np.clip(imgs_np * 0.5 + 0.5, 0, 1)
        imgs_np = (imgs_np * 255).astype(np.uint8).transpose(0, 2, 3, 1)  # [B, H, W, C]

        # discrete colormaps
        n_cls = self.num_nuclei_classes
        type_cmap = plt.cm.get_cmap('tab10', n_cls)
        type_norm = mcolors.BoundaryNorm(np.arange(n_cls + 1) - 0.5, n_cls)

        n_tis = self.num_tissue_classes
        tissue_cmap = plt.cm.get_cmap('tab10', n_tis)
        tissue_norm = mcolors.BoundaryNorm(np.arange(n_tis + 1) - 0.5, n_tis)

        num_cols = 6 if has_tissue else 5
        wandb_images = []
        for i in range(n_samples):
            fig, axes = plt.subplots(2, num_cols, figsize=(4 * num_cols, 8))
            name = names[i] if i < len(names) else str(i)
            fig.suptitle(f"Sample: {name}", fontsize=10)

            for row_idx, row_label in enumerate(["GT", "Pred"]):
                binary  = gt_binary[i]  if row_idx == 0 else pred_binary[i]
                typ     = gt_type[i]    if row_idx == 0 else pred_type[i]
                hv_h    = gt_hv[i, 0]   if row_idx == 0 else pred_hv[i, 0]
                hv_v    = gt_hv[i, 1]   if row_idx == 0 else pred_hv[i, 1]

                axes[row_idx, 0].imshow(imgs_np[i])
                axes[row_idx, 0].set_title(f"{row_label} Input", fontsize=8)
                axes[row_idx, 1].imshow(binary, cmap='gray', vmin=0, vmax=1)
                axes[row_idx, 1].set_title(f"{row_label} Binary", fontsize=8)
                axes[row_idx, 2].imshow(typ, cmap=type_cmap, norm=type_norm, interpolation='nearest')
                axes[row_idx, 2].set_title(f"{row_label} Type", fontsize=8)
                axes[row_idx, 3].imshow(hv_h, cmap='RdBu', vmin=-1, vmax=1)
                axes[row_idx, 3].set_title(f"{row_label} HV_H", fontsize=8)
                axes[row_idx, 4].imshow(hv_v, cmap='RdBu', vmin=-1, vmax=1)
                axes[row_idx, 4].set_title(f"{row_label} HV_V", fontsize=8)

                if has_tissue:
                    tis = gt_tissue[i] if row_idx == 0 else pred_tissue[i]
                    axes[row_idx, 5].imshow(tis, cmap=tissue_cmap, norm=tissue_norm, interpolation='nearest')
                    axes[row_idx, 5].set_title(f"{row_label} Tissue", fontsize=8)

                for col_idx in range(num_cols):
                    axes[row_idx, col_idx].axis('off')

            plt.tight_layout()
            wandb_images.append(wandb.Image(fig, caption=name))
            plt.close(fig)

        return {"images": wandb_images}

    def check_early_stopping(self, val_metrics: Dict[str, float]) -> bool:
        if self.early_stopping is None:
            return False
        return self.early_stopping.step(val_metrics['loss'])

    def save_checkpoint(self, path: str, epoch: int, val_metrics: Dict[str, float], best_pq: float = 0.0, best_epoch: int = 0, run_id: str = None):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_metrics': val_metrics,
            'best_pq': best_pq,
            'best_epoch': best_epoch,
        }
        if run_id is not None:
            checkpoint['run_id'] = run_id

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.early_stopping is not None:
            checkpoint['early_stopping_state'] = {
                'counter': self.early_stopping.counter,
                'best_score': self.early_stopping.best_score,
                'should_stop': self.early_stopping.should_stop,
            }

        tmp_path = path + ".tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"Scheduler state restored")

        if self.early_stopping is not None and 'early_stopping_state' in checkpoint:
            es_state = checkpoint['early_stopping_state']
            self.early_stopping.counter = es_state['counter']
            self.early_stopping.best_score = es_state['best_score']
            self.early_stopping.should_stop = es_state['should_stop']
            print(f"Early stopping state restored (counter={es_state['counter']}, best={es_state['best_score']})")

        epoch = checkpoint.get('epoch', 0)
        best_pq = checkpoint.get('best_pq', 0.0)
        best_epoch = checkpoint.get('best_epoch', 0)
        print(f"Checkpoint loaded from {path} (epoch {epoch})")

        return epoch + 1, best_pq, best_epoch  
