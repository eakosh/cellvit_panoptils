import os
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2

from data.transforms import get_tissue_context_aug, get_tissue_context_aug_strong
from utils.hv_map import gen_instance_hv_map


_tissue_context_aug_strong = get_tissue_context_aug_strong()


@dataclass
class PanopTILsPaths:
    root: str
    subset: str = "tcga"


class PanopTILsDataset(Dataset):
    PATCH_SIZE = 256

    def __init__(
        self,
        paths: PanopTILsPaths,
        file_list: Optional[List[str]] = None,
        transforms: Optional[Callable] = None,
        cache_dataset: bool = False,
        include_tissue_label: bool = True,
        cache_hv_maps: bool = False,
        apply_tissue_aug: bool = False,
        unlabeled_class: Optional[int] = None,
        background_class: Optional[int] = None,
        excluded_nuclei_classes: Optional[Set[int]] = None,
        excluded_tissue_classes: Optional[List[int]] = None,
        tissue_context_size: int = 512,
        tissue_strong_aug: bool = False,
        tissue_remap: Optional[Dict[int, int]] = None,
    ):
        self.paths = paths
        self.transforms = transforms
        self.cache_dataset = cache_dataset
        self.include_tissue_label = include_tissue_label
        self.cache_hv_maps = cache_hv_maps
        self.apply_tissue_aug = apply_tissue_aug
        self.unlabeled_class = unlabeled_class
        self.background_class = background_class
        self._excluded_nuclei_classes: Set[int] = set(excluded_nuclei_classes) if excluded_nuclei_classes else set()
        self._excluded_tissue_classes: Set[int] = set(excluded_tissue_classes) if excluded_tissue_classes else set()
        self.tissue_context_size = tissue_context_size
        self.tissue_strong_aug = tissue_strong_aug
        self.tissue_remap = tissue_remap
        self._tissue_remap_lut = None
        if self.tissue_remap:
            max_old = max(int(k) for k in self.tissue_remap.keys())
            self._tissue_remap_lut = np.zeros(max_old + 1, dtype=np.int32)
            for old, new in self.tissue_remap.items():
                self._tissue_remap_lut[int(old)] = int(new)

        self._hv_cache: Dict[int, np.ndarray] = {}

        self.rgb_dir = os.path.join(paths.root, paths.subset, "rgbs")
        self.mask_dir = os.path.join(paths.root, paths.subset, "masks")
        self.csv_dir = os.path.join(paths.root, paths.subset, "csv")

        if file_list is None:
            self.files = sorted([
                f for f in os.listdir(self.rgb_dir)
                if f.lower().endswith(".png")
            ])
        else:
            self.files = file_list

        if len(self.files) == 0:
            raise RuntimeError(f"No png files found in {self.rgb_dir}")

        # assume fixed tile size 1024x1024
        self.tiles_per_side = 1024 // self.PATCH_SIZE
        self.patches_per_image = self.tiles_per_side ** 2
       
        self.meta = {}
        self.tissue_pixel_counts: Counter = Counter()
        if self.include_tissue_label:
            for f in self.files:
                base = os.path.splitext(f)[0]
                mask_path = os.path.join(self.mask_dir, base + ".png")

                if os.path.exists(mask_path):
                    try:
                        mask = np.array(Image.open(mask_path))

                        tissue_mask = mask[:, :, 0].astype(np.int32)
                        if self._tissue_remap_lut is not None:
                            tissue_mask = self._tissue_remap_lut[tissue_mask]
                        nuclei_type_mask = mask[:, :, 1]

                        vals, counts = np.unique(tissue_mask, return_counts=True)
                        for v, c in zip(vals, counts):
                            self.tissue_pixel_counts[int(v)] += int(c)

                        patch_classes = []
                        patch_tissue_classes = []
                        for patch_idx in range(self.patches_per_image):
                            py = patch_idx // self.tiles_per_side
                            px = patch_idx % self.tiles_per_side
                            y0 = py * self.PATCH_SIZE
                            x0 = px * self.PATCH_SIZE

                            patch_nuc = nuclei_type_mask[y0:y0 + self.PATCH_SIZE, x0:x0 + self.PATCH_SIZE]
                            classes = {int(c) for c in np.unique(patch_nuc)
                                       if int(c) not in self._excluded_nuclei_classes}
                            patch_classes.append(classes)

                            patch_tis = tissue_mask[y0:y0 + self.PATCH_SIZE, x0:x0 + self.PATCH_SIZE]
                            u_t, c_t = np.unique(patch_tis, return_counts=True)
                            valid_t = ~np.isin(u_t, list(self._excluded_tissue_classes))
                            if np.any(valid_t):
                                patch_tissue_class = int(u_t[valid_t][c_t[valid_t].argmax()])
                            else:
                                patch_tissue_class = 0
                            patch_tissue_classes.append(patch_tissue_class)

                        self.meta[f] = {
                            "patch_classes": patch_classes,
                            "patch_tissue_classes": patch_tissue_classes,
                        }
                    except Exception:
                        self.meta[f] = {
                            "patch_classes": [set() for _ in range(self.patches_per_image)],
                            "patch_tissue_classes": [0] * self.patches_per_image,
                        }
                else:
                    self.meta[f] = {
                        "patch_classes": [set() for _ in range(self.patches_per_image)],
                        "patch_tissue_classes": [0] * self.patches_per_image,
                    }

        self._cached = set()
        self._cache_imgs: Dict[int, np.ndarray] = {}
        self._cache_tissue_masks: Dict[int, np.ndarray] = {}
        self._cache_nuclei_masks: Dict[int, np.ndarray] = {}
        self._cache_instance_maps: Dict[int, np.ndarray] = {}
        self._cache_tissue_context: Dict[int, np.ndarray] = {}
        self._cache_tissue_mask_context: Dict[int, np.ndarray] = {}

        if self.cache_dataset:
            print(f"Preloading {len(self.files)} images into memory")
            for i in range(len(self.files)):
                self._cache_imgs[i] = self._load_image(i)
                tissue, nuclei = self._load_mask(i)
                self._cache_tissue_masks[i] = tissue
                self._cache_nuclei_masks[i] = nuclei
                self._cache_instance_maps[i] = self._load_instance_map_from_csv(i, tissue.shape)
                self._cache_tissue_context[i] = cv2.resize(
                    self._cache_imgs[i], (self.tissue_context_size, self.tissue_context_size),
                    interpolation=cv2.INTER_AREA,
                )
                self._cache_tissue_mask_context[i] = cv2.resize(
                    tissue.astype(np.uint8), (self.tissue_context_size, self.tissue_context_size),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.int32)
                self._cached.add(i)
            print("Preloading complete")

    def __len__(self) -> int:
        return len(self.files) * self.patches_per_image

    def _load_image(self, img_idx: int) -> np.ndarray:
        name = self.files[img_idx]
        img_path = os.path.join(self.rgb_dir, name)
        return np.array(Image.open(img_path).convert("RGB")).astype(np.uint8)

    def _load_mask(self, img_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        name = self.files[img_idx]
        base = os.path.splitext(name)[0]
        mask_path = os.path.join(self.mask_dir, base + ".png")
        mask = np.array(Image.open(mask_path))
        
        # channel 0: tissue 
        # channel 1: nuclei 
        tissue_mask = mask[:, :, 0].astype(np.int32)
        nuclei_mask = mask[:, :, 1].astype(np.int32)

        if self._tissue_remap_lut is not None:
            tissue_mask = self._tissue_remap_lut[tissue_mask]

        return tissue_mask, nuclei_mask

    def _load_instance_map_from_csv(self, img_idx: int, img_shape: Tuple[int, int]) -> np.ndarray:
        name = self.files[img_idx]
        base = os.path.splitext(name)[0]
        csv_path = os.path.join(self.csv_dir, base + ".csv")

        instance_map = np.zeros(img_shape, dtype=np.int32)

        if not os.path.exists(csv_path):
            return instance_map

        try:
            df = pd.read_csv(csv_path)

            for inst_id, row in enumerate(df.itertuples(), start=1):

                coords_x = [int(x) for x in str(row.coords_x).split(',')]
                coords_y = [int(y) for y in str(row.coords_y).split(',')]

                points = np.array([[x, y] for x, y in zip(coords_x, coords_y)], dtype=np.int32)

                cv2.fillPoly(instance_map, [points], inst_id)
        except Exception:
            pass

        return instance_map

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], str]:
        img_idx = idx // self.patches_per_image
        patch_idx = idx % self.patches_per_image

        py = patch_idx // self.tiles_per_side
        px = patch_idx % self.tiles_per_side

        y0 = py * self.PATCH_SIZE
        x0 = px * self.PATCH_SIZE
        y1 = y0 + self.PATCH_SIZE
        x1 = x0 + self.PATCH_SIZE

        if self.cache_dataset and img_idx in self._cached:
            img = self._cache_imgs[img_idx]
            tissue_mask = self._cache_tissue_masks[img_idx]
            nuclei_mask = self._cache_nuclei_masks[img_idx]
            instance_map_full = self._cache_instance_maps[img_idx]
        else:
            img = self._load_image(img_idx)
            tissue_mask, nuclei_mask = self._load_mask(img_idx)
            instance_map_full = self._load_instance_map_from_csv(img_idx, tissue_mask.shape)
            if self.cache_dataset:
                self._cache_imgs[img_idx] = img
                self._cache_tissue_masks[img_idx] = tissue_mask
                self._cache_nuclei_masks[img_idx] = nuclei_mask
                self._cache_instance_maps[img_idx] = instance_map_full
                self._cached.add(img_idx)

        img = img[y0:y1, x0:x1]
        tissue_mask = tissue_mask[y0:y1, x0:x1]
        nuclei_mask = nuclei_mask[y0:y1, x0:x1]
        instance_map = instance_map_full[y0:y1, x0:x1].copy()

        unique_ids = np.unique(instance_map)
        unique_ids = unique_ids[unique_ids != 0] 
        if len(unique_ids) > 0:
            lut = np.zeros(unique_ids.max() + 1, dtype=np.int32)
            for new_id, old_id in enumerate(unique_ids, start=1):
                lut[old_id] = new_id
            mask = instance_map > 0
            instance_map[mask] = lut[instance_map[mask]]

        if (self.cache_dataset and img_idx in self._cached 
            and img_idx in self._cache_tissue_context):
            tissue_context = self._cache_tissue_context[img_idx].copy()
            tissue_mask_context = self._cache_tissue_mask_context[img_idx].copy()
        else:
            full_img = (self._cache_imgs[img_idx] 
                        if (self.cache_dataset and img_idx in self._cached) 
                        else self._load_image(img_idx))
            tissue_context = cv2.resize(full_img, 
                                        (self.tissue_context_size, self.tissue_context_size), 
                                        interpolation=cv2.INTER_AREA)
            full_tissue = (self._cache_tissue_masks[img_idx] 
                           if (self.cache_dataset and img_idx in self._cached) 
                           else self._load_mask(img_idx)[0])
            tissue_mask_context = cv2.resize(full_tissue.astype(np.uint8),
                                          (self.tissue_context_size, self.tissue_context_size),
                                          interpolation=cv2.INTER_NEAREST).astype(np.int32)

        nuclei_type_map = nuclei_mask.copy()
        if self.background_class is not None:
            nuclei_type_map[instance_map == 0] = self.background_class
        if self.unlabeled_class is not None and self._excluded_nuclei_classes:
            untyped = (instance_map > 0) & np.isin(nuclei_mask, list(self._excluded_nuclei_classes))
            nuclei_type_map[untyped] = self.unlabeled_class

        if self.transforms is not None:
            out = self.transforms(
                image=img,
                masks=[tissue_mask, instance_map, nuclei_type_map],
            )
            img = out["image"]
            m = out["masks"]
            tissue_mask = m[0]
            instance_map = m[1]
            nuclei_type_map = m[2]

        nuclei_binary_map = (instance_map > 0).astype(np.int32)

        if self.cache_hv_maps and idx in self._hv_cache:
            hv_map = self._hv_cache[idx]
        else:
            hv_map = gen_instance_hv_map(instance_map)
            if self.cache_hv_maps:
                self._hv_cache[idx] = hv_map

        if self.transforms is not None:
            img_t = torch.from_numpy(img).float().permute(2, 0, 1)
        else:
            # normalization (mean=0.5, std=0.5)
            img_t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            img_t = (img_t - 0.5) / 0.5

        if self.apply_tissue_aug:
            if self.tissue_strong_aug:
                aug = _tissue_context_aug_strong(
                    image=tissue_context,
                    mask=tissue_mask_context.astype(np.int32),
                )
                tissue_context = aug["image"]
                tissue_mask_context = aug["mask"].astype(np.int32)
            else:
                tissue_context, tissue_mask_context, py, px = get_tissue_context_aug(
                    tissue_context, tissue_mask_context, py, px, self.tiles_per_side,
                )
        tissue_context_t = torch.from_numpy(tissue_context).float().permute(2, 0, 1) / 255.0
        tissue_context_t = (tissue_context_t - 0.5) / 0.5

        targets = {
            "tissue_mask": torch.from_numpy(np.ascontiguousarray(tissue_mask)).long(),
            "tissue_mask_context": torch.from_numpy(np.ascontiguousarray(tissue_mask_context)).long(),
            "instance_map": torch.from_numpy(np.ascontiguousarray(instance_map)).long(),
            "nuclei_type_map": torch.from_numpy(np.ascontiguousarray(nuclei_type_map)).long(),
            "nuclei_binary_map": torch.from_numpy(np.ascontiguousarray(nuclei_binary_map)).long(),
            "hv_map": torch.from_numpy(np.ascontiguousarray(hv_map)).float(),
            "tissue_context": tissue_context_t,
            "crop_coords": torch.tensor([py, px], dtype=torch.long),
            "patch_idx": torch.tensor(patch_idx, dtype=torch.long),
        }

        name = self.files[img_idx]
        return img_t, targets, name
