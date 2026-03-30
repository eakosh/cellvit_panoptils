from collections import OrderedDict

import torch
import torch.nn as nn

from model.cellvit import CellViT
from model.model_utils import Conv2DBlock


class CellViTWithTissue(CellViT):
    def __init__(self, tissue_fusion: bool = False,
                 use_compatibility_constraint: bool = False,
                 nuclei_tissue_compatibility: dict = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.tissue_fusion = tissue_fusion
        self.use_compatibility_constraint = use_compatibility_constraint

        self.tissue_segmentation_decoder = self.create_upsampling_branch(
            self.num_tissue_classes
        )

        if self.tissue_fusion:
            # replace nuclei_type decoder0_header to accept 128 (nuclei) + 128 (tissue) channels
            self.nuclei_type_maps_decoder.decoder0_header = nn.Sequential(
                Conv2DBlock(64 * 2 + 128, 64, dropout=self.drop_rate),
                Conv2DBlock(64, 64, dropout=self.drop_rate),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=self.num_nuclei_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
            )

        if self.use_compatibility_constraint:
            # fixed binary mask: which (nucleus, tissue) pairs are allowed
            C = torch.zeros(self.num_nuclei_classes, self.num_tissue_classes)
            for nuc_cls, tissue_list in (nuclei_tissue_compatibility or {}).items():
                for t in tissue_list:
                    C[int(nuc_cls), t] = 1.0
            self.register_buffer('compat_mask', C)
            # learned kernel: per-pair weight, masked by fixed binary
            self.compat_kernel = nn.Parameter(torch.rand_like(C))


    def _apply_compatibility_constraint(self, nt_logits, ts_logits):
        kernel = self.compat_kernel * self.compat_mask  # (N, T)
        attention = torch.einsum('nt,bthw->bnhw', kernel, ts_logits) # (B, N, H, W)
        return nt_logits * attention

    def _forward_upsample_with_features(
        self, z0, z1, z2, z3, z4, branch_decoder
    ):
        """Same as _forward_upsample but also returns pre-header features"""
        b4 = branch_decoder.bottleneck_upsampler(z4)
        b3 = self.decoder3(z3)
        b3 = branch_decoder.decoder3_upsampler(torch.cat([b3, b4], dim=1))
        b2 = self.decoder2(z2)
        b2 = branch_decoder.decoder2_upsampler(torch.cat([b2, b3], dim=1))
        b1 = self.decoder1(z1)
        b1 = branch_decoder.decoder1_upsampler(torch.cat([b1, b2], dim=1))
        b0 = self.decoder0(z0)
        features = torch.cat([b0, b1], dim=1) # (B, 128, H, W)
        output = branch_decoder.decoder0_header(features)
        return output, features

    def _forward_upsample_fused(
        self, z0, z1, z2, z3, z4, branch_decoder, extra_features
    ):
        """Forward upsample that appends extra_features at the decoder0_header stage"""
        b4 = branch_decoder.bottleneck_upsampler(z4)
        b3 = self.decoder3(z3)
        b3 = branch_decoder.decoder3_upsampler(torch.cat([b3, b4], dim=1))
        b2 = self.decoder2(z2)
        b2 = branch_decoder.decoder2_upsampler(torch.cat([b2, b3], dim=1))
        b1 = self.decoder1(z1)
        b1 = branch_decoder.decoder1_upsampler(torch.cat([b1, b2], dim=1))
        b0 = self.decoder0(z0)
        # 64 (b0) + 64 (b1) + 128 (tissue_features) = 256 channels
        output = branch_decoder.decoder0_header(
            torch.cat([b0, b1, extra_features], dim=1)
        )
        return output

    
    def forward(self, x: torch.Tensor, retrieve_tokens: bool = False) -> dict:
        assert x.shape[-2] % self.patch_size == 0
        assert x.shape[-1] % self.patch_size == 0

        out_dict = {}

        classifier_logits, _, z = self.encoder(x)
        out_dict["tissue_types"] = classifier_logits

        z0, z1, z2, z3, z4 = x, *z

        patch_dim = [int(d / self.patch_size) for d in [x.shape[-2], x.shape[-1]]]
        z4 = z4[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z3 = z3[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z2 = z2[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z1 = z1[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)

        # binary and HV branches
        if self.regression_loss:
            nb_map = self._forward_upsample(
                z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder
            )
            out_dict["nuclei_binary_map"] = nb_map[:, :2, :, :]
            out_dict["regression_map"] = nb_map[:, 2:, :, :]
        else:
            out_dict["nuclei_binary_map"] = self._forward_upsample(
                z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder
            )
        out_dict["hv_map"] = self._forward_upsample(
            z0, z1, z2, z3, z4, self.hv_map_decoder
        )

        # tissue segmentation branch 
        tissue_map, tissue_features = self._forward_upsample_with_features(
            z0, z1, z2, z3, z4, self.tissue_segmentation_decoder
        )
        out_dict["tissue_segmentation_map"] = tissue_map

        # nuclei type branch
        if self.tissue_fusion:
            nt_logits = self._forward_upsample_fused(
                z0, z1, z2, z3, z4, self.nuclei_type_maps_decoder, tissue_features
            )
        else:
            nt_logits = self._forward_upsample(
                z0, z1, z2, z3, z4, self.nuclei_type_maps_decoder
            )

        # apply biological compatibility constraint
        if self.use_compatibility_constraint:
            out_dict["nuclei_type_map_pre"] = nt_logits
            nt_logits = self._apply_compatibility_constraint(nt_logits, tissue_map)

        out_dict["nuclei_type_map"] = nt_logits

        if retrieve_tokens:
            out_dict["tokens"] = z4

        return out_dict
