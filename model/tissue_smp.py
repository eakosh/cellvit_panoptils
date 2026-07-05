# Copyright © 2025 Olena Kosharova, FIIT STU.
# Part of the "Tissue-Context CellViT Extension" (bachelor's thesis, FIIT STU).
# Licensed under the Apache License 2.0 with the Commons Clause restriction.
# See the LICENSE file in the project root for full terms.

import importlib
import subprocess
import sys

import torch
import torch.nn as nn


def _ensure_smp():
    try:
        import segmentation_models_pytorch  
        return
    except ImportError:
        pass
    print("Installing segmentation_models_pytorch and timm")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--no-cache-dir",
        "timm>=0.9.0", "segmentation-models-pytorch>=0.3.3",
    ])
    importlib.invalidate_caches()


_ensure_smp()
import segmentation_models_pytorch as smp  


class SMPSegEncoder(nn.Module):
    def __init__(
        self,
        encoder_name: str = "mit_b2",
        encoder_weights: str = "imagenet",
        num_classes: int = 9,
        decoder: str = "unet",
        in_channels: int = 3,
    ):
        super().__init__()
        decoder = decoder.lower()
        if decoder == "unet":
            model_cls = smp.Unet
        else:
            raise ValueError(f"Unsupported decoder: {decoder}")

        self.model = model_cls(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )

    def forward(self, x):
        logits = self.model(x)
        return logits, None

    def freeze_encoder(self):
        for p in self.model.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.model.encoder.parameters():
            p.requires_grad = True
