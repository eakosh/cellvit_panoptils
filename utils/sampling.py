# Copyright © 2025 Olena Kosharova, FIIT STU.
# Part of the "Tissue-Context CellViT Extension" (bachelor's thesis, FIIT STU).
# Licensed under the Apache License 2.0 with the Commons Clause restriction.
# See the LICENSE file in the project root for full terms.

from collections import Counter
from typing import Dict, List, Optional, Set
import numpy as np
import torch
from torch.utils.data import Sampler


def compute_sampling_weights(
    patch_tissue_classes: List[int],
    patch_nuclei_classes: List[Set[int]],
    gamma_s: float = 0.85,
    ignore_nuclei_classes: Optional[Set[int]] = None,
    patches_per_image: int = 16,
    tissue_dedup_boost: float = 3.0,
) -> torch.Tensor:

    if ignore_nuclei_classes is None:
        ignore_nuclei_classes = set()

    n_patches = len(patch_tissue_classes)
    assert len(patch_nuclei_classes) == n_patches, "Length mismatch"

    # tissue weights
    tissue_count = Counter(patch_tissue_classes)
    w_tissue = np.array([n_patches / (gamma_s * tissue_count[ct] + (1.0 - gamma_s) * n_patches)
                        for ct in patch_tissue_classes], dtype=np.float32)
    w_tissue_max = w_tissue.max()
    w_tissue_norm = w_tissue / w_tissue_max if w_tissue_max > 0 else np.ones(n_patches, dtype=np.float32)

    # cell weights
    filtered = [classes - ignore_nuclei_classes for classes in patch_nuclei_classes]
    n_cells = sum(len(c) for c in filtered)

    if n_cells == 0 or gamma_s == 0.0:
        w_cell_norm = np.ones(n_patches, dtype=np.float32)
    else:
        class_patch_count = Counter()
        for classes in filtered:
            for c in classes:
                class_patch_count[c] += 1

        class_factor = {j: n_cells / (gamma_s * n_j + (1.0 - gamma_s) * n_cells)
                for j, n_j in class_patch_count.items()}

        w_cell = np.array([(1.0 - gamma_s) + gamma_s * sum(class_factor[j] 
                        for j in classes) for classes in filtered], dtype=np.float32)
        w_cell_max = w_cell.max()
        w_cell_norm = w_cell / w_cell_max if w_cell_max > 0 else np.ones(n_patches, dtype=np.float32)

    p = w_tissue_norm + w_cell_norm

    # boost patches with tissue
    if tissue_dedup_boost > 1.0 and patches_per_image > 1:
        for i in range(0, n_patches, patches_per_image):
            p[i] *= tissue_dedup_boost

    return torch.tensor(p, dtype=torch.float32)


class PatchWeightedRandomSampler(Sampler):
    def __init__(self, weights: torch.Tensor, patches_per_image: int, 
                 num_samples: int, replacement: bool = True):
        self.weights = weights.float()
        self.patches_per_image = patches_per_image
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        indices = torch.multinomial(
            self.weights, self.num_samples, replacement=self.replacement
        )
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples
