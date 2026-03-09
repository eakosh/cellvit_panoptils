from dataclasses import dataclass
from typing import Optional
from torch.utils.data import DataLoader

from datasets.panoptils import PanopTILsDataset, PanopTILsPaths
from utils.sampling import (
    compute_sampling_weights,
    PatchWeightedRandomSampler,
)


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
    unlabeled_class: Optional[int] = None
    background_nuclei_class: Optional[int] = None  


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

        self.train_ds = PanopTILsDataset(
            paths=paths,
            file_list=self.train_files,
            transforms=self.train_transforms,
            cache_dataset=self.cfg.cache_dataset,
            include_tissue_label=True,
            unlabeled_class=self.cfg.unlabeled_class,
        )
        self.val_ds = PanopTILsDataset(
            paths=paths,
            file_list=self.val_files,
            transforms=self.val_transforms,
            cache_dataset=self.cfg.cache_dataset,
            include_tissue_label=True,
            cache_hv_maps=True,
            unlabeled_class=self.cfg.unlabeled_class,
        )

        self._train_loader = None
        self._val_loader = None

    def train_dataloader(self):
        if self._train_loader is not None:
            return self._train_loader

        if self.cfg.use_weighted_sampler:
            patch_tissue_classes = []
            patch_nuclei_classes = []
            for f in self.train_ds.files:
                meta = self.train_ds.meta[f]
                patch_tissue_classes.extend(meta["patch_tissue_classes"])
                patch_nuclei_classes.extend(meta["patch_classes"])

            ignore: set = set()
            if self.cfg.unlabeled_class is not None:
                ignore.add(self.cfg.unlabeled_class)       
            if self.cfg.background_nuclei_class is not None:
                ignore.add(self.cfg.background_nuclei_class)  

            weights = compute_sampling_weights(
                patch_tissue_classes=patch_tissue_classes,
                patch_nuclei_classes=patch_nuclei_classes,
                gamma_s=self.cfg.gamma_s,
                ignore_nuclei_classes=ignore,
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
            prefetch_factor=4 if self.cfg.num_workers > 0 else None,
            drop_last=True,
        )
        return self._train_loader

    def val_dataloader(self):
        if self._val_loader is not None:
            return self._val_loader

        self._val_loader = DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers,
            prefetch_factor=4 if self.cfg.num_workers > 0 else None,
        )
        return self._val_loader
