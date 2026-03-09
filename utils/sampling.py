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
) -> torch.Tensor:
    
    if ignore_nuclei_classes is None:
        ignore_nuclei_classes = set()

    N_train = len(patch_tissue_classes)
    assert len(patch_nuclei_classes) == N_train, "Length mismatch"

    # tissue weights
    tissue_count = Counter(patch_tissue_classes)
    w_tissue = np.array(
        [N_train / (gamma_s * tissue_count[ct] + (1.0 - gamma_s) * N_train)
         for ct in patch_tissue_classes],
        dtype=np.float32,
    )
    w_tissue_max = w_tissue.max()
    w_tissue_norm = w_tissue / w_tissue_max if w_tissue_max > 0 else np.ones(N_train, dtype=np.float32)

    # сell weights
    filtered = [classes - ignore_nuclei_classes for classes in patch_nuclei_classes]
    N_cell = sum(len(c) for c in filtered)

    if N_cell == 0 or gamma_s == 0.0:
        w_cell_norm = np.ones(N_train, dtype=np.float32)
    else:
        class_patch_count = Counter()
        for classes in filtered:
            for c in classes:
                class_patch_count[c] += 1

        class_factor = {
            j: N_cell / (gamma_s * n_j + (1.0 - gamma_s) * N_cell)
                for j, n_j in class_patch_count.items()
        }

        w_cell = np.array(
            [(1.0 - gamma_s) + gamma_s * sum(class_factor[j] for j in classes)
                                                    for classes in filtered],
            dtype=np.float32,
        )
        w_cell_max = w_cell.max()
        w_cell_norm = w_cell / w_cell_max if w_cell_max > 0 else np.ones(N_train, dtype=np.float32)

    p = w_tissue_norm + w_cell_norm
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
