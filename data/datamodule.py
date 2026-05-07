from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Set
import numpy as np
from torch.utils.data import DataLoader

from datasets.panoptils import PanopTILsDataset, PanopTILsPaths
from utils.sampling import compute_sampling_weights, PatchWeightedRandomSampler
from data.constants import TISSUE_REMAP, NUCLEI_TISSUE_COMPATIBILITY


@dataclass
class DataConfig:
    root: str
    subset: str = "tcga"
    batch_size: int = 2
    num_workers: int = 4
    cache_dataset: bool = False
    pin_memory: bool = True
    persistent_workers: bool = True
    use_weighted_sampler: bool = True
    gamma_s: float = 0.85
    num_nuclei_classes: int = 10
    num_tissue_classes: int = 9
    nuclei_unlabeled_class: Optional[int] = None
    nuclei_background_class: Optional[int] = None
    nuclei_ambiguous_classes: List[int] = field(default_factory=list)
    tissue_ignore_classes: List[int] = field(default_factory=lambda: [0])
    excluded_tissue_classes: List[int] = field(default_factory=lambda: [0, 8])
    background_loss_weight: float = 0.3
    nuclei_tissue_compatibility: Optional[dict] = field(default_factory=lambda: dict(NUCLEI_TISSUE_COMPATIBILITY))
    tissue_remap: Optional[dict] = field(default_factory=lambda: dict(TISSUE_REMAP))
    tissue_context_size: int = 512
    tissue_strong_aug: bool = False

    # union of classes that are not used as real nucleus types
    excluded_nuclei_classes: Set[int] = field(init=False)

    def __post_init__(self):
        s = set(self.nuclei_ambiguous_classes)
        if self.nuclei_unlabeled_class is not None:
            s.add(self.nuclei_unlabeled_class)
        if self.nuclei_background_class is not None:
            s.add(self.nuclei_background_class)
        self.excluded_nuclei_classes = s


class PanopTILsDataModule:
    def __init__(self, cfg: DataConfig, train_files=None, val_files=None,
                 train_transforms=None, val_transforms=None):
        self.cfg = cfg
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.train_files = train_files
        self.val_files = val_files

    def setup(self):
        paths = PanopTILsPaths(root=self.cfg.root, subset=self.cfg.subset)

        ds_kwargs = dict(
            unlabeled_class=self.cfg.nuclei_unlabeled_class,
            background_class=self.cfg.nuclei_background_class,
            excluded_nuclei_classes=self.cfg.excluded_nuclei_classes,
            excluded_tissue_classes=self.cfg.excluded_tissue_classes,
            tissue_context_size=self.cfg.tissue_context_size,
            tissue_strong_aug=self.cfg.tissue_strong_aug,
            tissue_remap=self.cfg.tissue_remap,
        )

        if self.train_files:
            self.train_ds = PanopTILsDataset(
                paths=paths,
                file_list=self.train_files,
                transforms=self.train_transforms,
                cache_dataset=self.cfg.cache_dataset,
                include_tissue_label=True,
                apply_tissue_aug=True,
                **ds_kwargs,
            )
        else:
            self.train_ds = None

        if self.val_files:
            self.val_ds = PanopTILsDataset(
                paths=paths,
                file_list=self.val_files,
                transforms=self.val_transforms,
                cache_dataset=self.cfg.cache_dataset,
                include_tissue_label=True,
                cache_hv_maps=True,
                apply_tissue_aug=False,
                **ds_kwargs,
            )
        else:
            self.val_ds = None

        self._patch_tissue_classes: List[int] = []
        self._patch_nuclei_classes: List[Set[int]] = []
        if self.train_ds is not None:
            for f in self.train_ds.files:
                meta = self.train_ds.meta[f]
                self._patch_tissue_classes.extend(meta["patch_tissue_classes"])
                self._patch_nuclei_classes.extend(meta["patch_classes"])

        self._train_loader = None
        self._val_loader = None

    def nt_class_weights(self, num_nuclei_classes: int) -> List[float]:
        background = self.cfg.nuclei_background_class
        excluded = self.cfg.excluded_nuclei_classes
        nucleus_classes = [c for c in range(num_nuclei_classes) if c not in excluded]

        counts = Counter()
        for class_set in self._patch_nuclei_classes:
            for c in class_set:
                if c in nucleus_classes:
                    counts[c] += 1

        nonzero = [counts[c] for c in nucleus_classes if counts[c] > 0]
        median_count = float(np.median(nonzero)) if nonzero else 1.0

        weights = [0.0] * num_nuclei_classes
        for c in nucleus_classes:
            weights[c] = median_count / counts[c] if counts[c] > 0 else median_count
        if background is not None:
            weights[background] = self.cfg.background_loss_weight
        return weights

    def ts_class_weights(self, num_tissue_classes: int,
                         ignore_classes: set) -> List[float]:
        counts = self.train_ds.tissue_pixel_counts
        active = [c for c in range(num_tissue_classes) if c not in ignore_classes]

        nonzero = [counts[c] for c in active if counts[c] > 0]
        median_count = float(np.median(nonzero)) if nonzero else 1.0

        weights = [0.0] * num_tissue_classes
        for c in active:
            weights[c] = median_count / counts[c] if counts[c] > 0 else median_count
        return weights

    def train_dataloader(self):
        if self._train_loader is not None:
            return self._train_loader

        if self.cfg.use_weighted_sampler:
            ignore = self.cfg.excluded_nuclei_classes

            weights = compute_sampling_weights(
                patch_tissue_classes=self._patch_tissue_classes,
                patch_nuclei_classes=self._patch_nuclei_classes,
                gamma_s=self.cfg.gamma_s,
                ignore_nuclei_classes=ignore,
                patches_per_image=self.train_ds.patches_per_image,
            )
            print(f"Oversampling: gamma_s={self.cfg.gamma_s}, "
                  f"ignore_nuclei={ignore}, N_patches={len(weights)}")

            sampler = PatchWeightedRandomSampler(
                weights,
                patches_per_image=self.train_ds.patches_per_image,
                num_samples=len(self.train_ds),
                replacement=True,
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True

        self._train_loader = DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers,
            prefetch_factor=2 if self.cfg.num_workers > 0 else None,
            drop_last=True,
        )
        return self._train_loader

    def val_dataloader(self):
        if self.val_ds is None:
            return []

        if self._val_loader is not None:
            return self._val_loader

        self._val_loader = DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers,
            prefetch_factor=2 if self.cfg.num_workers > 0 else None,
        )
        return self._val_loader
