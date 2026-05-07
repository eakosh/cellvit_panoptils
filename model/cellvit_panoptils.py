import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from model.cellvit import CellViT
from model.model_utils import Conv2DBlock


def apply_posthoc_tissue_constraint(pred_type_map: np.ndarray, pred_instances: np.ndarray,
                                    tissue_map: np.ndarray, nt_logits: np.ndarray, compat_map: dict) -> np.ndarray:
    tissue_to_nuclei: dict[int, set[int]] = {}
    for nuc_cls, tissue_list in compat_map.items():
        nuc_cls = int(nuc_cls)
        for t in tissue_list:
            t = int(t)
            if t not in tissue_to_nuclei:
                tissue_to_nuclei[t] = set()
            tissue_to_nuclei[t].add(nuc_cls)

    corrected = pred_type_map.copy()
    instance_ids = np.unique(pred_instances)
    instance_ids = instance_ids[instance_ids > 0]

    n_fixed = 0
    for inst_id in instance_ids:
        mask = pred_instances == inst_id
        ys, xs = np.where(mask)
        cy, cx = int(ys.mean()), int(xs.mean())

        nuc_type = int(pred_type_map[cy, cx])
        tissue_type = int(tissue_map[cy, cx])

        compatible_tissues = compat_map.get(nuc_type)
        if compatible_tissues is None:
            continue
        if tissue_type in [int(t) for t in compatible_tissues]:
            continue  

        allowed_nuclei = tissue_to_nuclei.get(tissue_type)
        if not allowed_nuclei:
            continue

        best_cls, best_logit = None, -float('inf')
        for cls in allowed_nuclei:
            if cls < nt_logits.shape[0]:
                logit = float(nt_logits[cls, cy, cx])
                if logit > best_logit:
                    best_logit = logit
                    best_cls = cls

        if best_cls is not None and best_cls != nuc_type:
            corrected[mask] = best_cls
            n_fixed += 1

    return corrected, n_fixed


class AFFBlock(nn.Module):
    def __init__(self, nuclei_channels: int, tissue_classes: int = 9, reduction: int = 4):
        super().__init__()
        self.tissue_proj = nn.Conv2d(tissue_classes, nuclei_channels, kernel_size=1, bias=True)
        inner = max(nuclei_channels // reduction, 4)
        self.local_attn = nn.Sequential(
            nn.Conv2d(nuclei_channels, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner, nuclei_channels, 1, bias=False),
            nn.BatchNorm2d(nuclei_channels),
        )
        self.global_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(nuclei_channels, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner, nuclei_channels, 1, bias=False),
            nn.BatchNorm2d(nuclei_channels),
        )

    def forward(self, nuclei_feat: torch.Tensor, tissue_probs: torch.Tensor) -> torch.Tensor:
        H, W = nuclei_feat.shape[-2:]
        if tissue_probs.shape[-2:] != (H, W):
            tissue_probs = F.interpolate(tissue_probs, size=(H, W), mode='bilinear', align_corners=False)
        tissue_proj = self.tissue_proj(tissue_probs)
        combined = nuclei_feat + tissue_proj
        w = torch.sigmoid(self.local_attn(combined) + self.global_attn(combined))
        return w * nuclei_feat + (1.0 - w) * tissue_proj


class MultiLevelAFF(nn.Module):
    def __init__(self, ch_b3: int = 256, ch_b2: int = 128, ch_b1: int = 64, 
                 tissue_classes: int = 9, reduction: int = 4):
        super().__init__()
        self.aff_b3 = AFFBlock(ch_b3, tissue_classes, reduction=reduction)
        self.aff_b2 = AFFBlock(ch_b2, tissue_classes, reduction=reduction)
        self.aff_b1 = AFFBlock(ch_b1, tissue_classes, reduction=reduction)


class BottleneckCrossAttention(nn.Module):
    def __init__(self, nuclei_channels: int, tissue_classes: int = 9, 
                 embed_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Conv2d(nuclei_channels, embed_dim, 1)
        self.k_proj = nn.Conv2d(tissue_classes, embed_dim, 1)
        self.v_proj = nn.Conv2d(tissue_classes, embed_dim, 1)
        self.out_proj = nn.Sequential(
            nn.Conv2d(embed_dim, nuclei_channels, 1),
            nn.BatchNorm2d(nuclei_channels),
            nn.Dropout2d(dropout),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(nuclei_channels * 2, 1, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.out_proj[0].weight)
        nn.init.zeros_(self.out_proj[0].bias)
        self.scale = embed_dim ** -0.5

    def forward(self, nuclei_feat: torch.Tensor, tissue_probs: torch.Tensor) -> torch.Tensor:
        B, _, H, W = nuclei_feat.shape
        if tissue_probs.shape[-2:] != (H, W):
            tissue_probs = F.interpolate(tissue_probs, size=(H, W), mode='bilinear', align_corners=False)
        Q = self.q_proj(nuclei_feat).flatten(2)
        K = self.k_proj(tissue_probs).flatten(2)
        V = self.v_proj(tissue_probs).flatten(2)
        attn = torch.bmm(Q.transpose(1, 2), K) * self.scale
        attn = torch.softmax(attn.float(), dim=-1).to(V.dtype)
        out = torch.bmm(V, attn.transpose(1, 2))
        out = out.view(B, self.embed_dim, H, W)
        out = self.out_proj(out)
        gate = self.gate(torch.cat([nuclei_feat, out], dim=1))
        return nuclei_feat + gate * out


class CellViTWithTissue(CellViT):
    def __init__(self, tissue_fusion: str = "none",
                 use_compatibility_constraint: bool = False,
                 nuclei_tissue_compatibility: dict = None,
                 fusion_warmup_epochs: int = 0,
                 freeze_tissue_after_fusion_warmup: bool = True,
                 tissue_encoder_type: str = "shared",
                 tissue_encoder_kwargs: dict = None,
                 fusion_embed_dim: int = 64,
                 fusion_reduction: int = 4,
                 **kwargs):
        super().__init__(**kwargs)
        self.tissue_fusion = tissue_fusion
        self.use_compatibility_constraint = use_compatibility_constraint
        self.fusion_warmup_epochs = fusion_warmup_epochs
        self.freeze_tissue_after_fusion_warmup = freeze_tissue_after_fusion_warmup
        self.tissue_encoder_type = tissue_encoder_type
        self._current_epoch = 0
        self._tissue_frozen = False

        if tissue_encoder_type == "smp":
            from model.tissue_smp import SMPSegEncoder
            kw = tissue_encoder_kwargs or {}
            self.tissue_encoder = SMPSegEncoder(
                encoder_name=kw.get("encoder_name", "mit_b2"),
                encoder_weights=kw.get("encoder_weights", "imagenet"),
                num_classes=self.num_tissue_classes,
                decoder=kw.get("decoder", "unet"),
            )
        elif tissue_encoder_type == "shared":
            self.tissue_segmentation_decoder = self.create_upsampling_branch(self.num_tissue_classes)

        if self.tissue_fusion == "multi_aff":
            self.multi_aff = MultiLevelAFF(
                ch_b3=256, ch_b2=128, ch_b1=64,
                tissue_classes=self.num_tissue_classes,
                reduction=fusion_reduction,
            )
        elif self.tissue_fusion == "cross_attn_bottleneck":
            bn_dim = self.nuclei_type_maps_decoder.bottleneck_upsampler.out_channels
            self.cross_attn_bn = BottleneckCrossAttention(
                nuclei_channels=bn_dim,
                tissue_classes=self.num_tissue_classes,
                embed_dim=fusion_embed_dim,
            )

        if self.use_compatibility_constraint:
            compat_mask = torch.zeros(self.num_nuclei_classes, self.num_tissue_classes)
            for nuc_cls, tissue_list in (nuclei_tissue_compatibility or {}).items():
                for t in tissue_list:
                    compat_mask[int(nuc_cls), t] = 1.0
            self.register_buffer('compat_mask', compat_mask)
            self.compat_kernel = nn.Parameter(torch.rand_like(compat_mask))


    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, '_keep_tissue_eval', False) and hasattr(self, 'tissue_encoder'):
            self.tissue_encoder.eval()
        return self


    def set_epoch(self, epoch: int):
        if (self._current_epoch < self.fusion_warmup_epochs and epoch >= self.fusion_warmup_epochs 
            and self.tissue_fusion != "none"):
            print(f"\nFusion warmup is complete. Enabling {self.tissue_fusion} fusion")
            if self.freeze_tissue_after_fusion_warmup and not self._tissue_frozen and hasattr(self, 'tissue_encoder'):
                for p in self.tissue_encoder.parameters():
                    p.requires_grad = False
                self._tissue_frozen = True
                print(f"Tissue encoder is frozen\n")
        self._current_epoch = epoch

    def _apply_compatibility_constraint(self, nt_logits, t_logits, nb_logits):
        orig_dtype = nt_logits.dtype
        nt_logits_f = nt_logits.float()
        t_logits_f = t_logits.float()

        kernel = F.softplus(self.compat_kernel) * self.compat_mask
        t_probs = torch.softmax(t_logits_f, dim=1)
        B, T, H, W = t_probs.shape
        t_flat = t_probs.view(B, T, -1)
        attn = torch.matmul(kernel, t_flat)
        attn = attn.view(B, -1, H, W)

        has_compat = (self.compat_mask.sum(dim=1) > 0).view(1, -1, 1, 1)
        scale = torch.where(has_compat, attn, torch.ones_like(attn))
        scale = scale.clamp(min=1e-4, max=10.0)

        nb_prob = torch.softmax(nb_logits.float(), dim=1)[:, 1:2]
        scale = nb_prob * scale + (1.0 - nb_prob)

        return (nt_logits_f * scale).to(orig_dtype)

    def _crop_and_upsample_tissue(self, tissue_full, crop_coords, target_size):
        B, C, H, W = tissue_full.shape
        tiles = 4  # 1024 / 256
        tile_h, tile_w = H // tiles, W // tiles  

        crops = []
        for i in range(B):
            py, px = crop_coords[i, 0].item(), crop_coords[i, 1].item()
            crop = tissue_full[i:i+1, :, py * tile_h:(py + 1) * tile_h, px * tile_w:(px + 1) * tile_w]
            crops.append(crop)
        cropped = torch.cat(crops, dim=0)

        if cropped.shape[-1] != target_size:
            cropped = F.interpolate(cropped, size=target_size,
                                    mode='bilinear', align_corners=False)
        return cropped

    def forward(self, x: torch.Tensor, tissue_context: torch.Tensor = None, crop_coords: torch.Tensor = None,
                retrieve_tokens: bool = False, oracle_tissue: torch.Tensor = None) -> dict:
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

        out_dict["nuclei_binary_map"] = self._forward_upsample(z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder)
        out_dict["hv_map"] = self._forward_upsample(z0, z1, z2, z3, z4, self.hv_map_decoder)

        # tissue branch
        if  self.tissue_encoder_type == "external" and oracle_tissue is not None:
            B, H, W = oracle_tissue.shape
            tissue_map = torch.full((B, self.num_tissue_classes, H, W), -1e4, dtype=x.dtype, device=x.device)
            tissue_map.scatter_(1, oracle_tissue.unsqueeze(1).long().clamp(0, self.num_tissue_classes - 1), 1e4)
            tissue_features = None
        elif self.tissue_encoder_type == "shared":
            tissue_features = self._forward_upsample_features(z0, z1, z2, z3, z4, self.tissue_segmentation_decoder)
            tissue_map = self.tissue_segmentation_decoder.decoder0_header(tissue_features)
        else:
            tissue_input = tissue_context if tissue_context is not None else x
            tissue_map_full, tissue_features_full = self.tissue_encoder(tissue_input)
            out_dict["tissue_segmentation_map_full"] = tissue_map_full
            if tissue_context is not None and crop_coords is not None:
                tissue_map = self._crop_and_upsample_tissue(tissue_map_full, crop_coords, target_size=x.shape[-1])
                if tissue_features_full is not None:
                    tissue_features = self._crop_and_upsample_tissue(tissue_features_full, crop_coords, target_size=x.shape[-1])
                else:
                    tissue_features = None
            else:
                tissue_map = tissue_map_full
                tissue_features = tissue_features_full

        out_dict["tissue_segmentation_map"] = tissue_map

        # nuclei type branch 
        if self.tissue_fusion == "multi_aff":
            tissue_probs = torch.softmax(tissue_map, dim=1)
            if self._current_epoch < self.fusion_warmup_epochs:
                tissue_probs = torch.zeros_like(tissue_probs)
            pre_header = self._forward_upsample_features_multi_aff(z0, z1, z2, z3, z4, self.nuclei_type_maps_decoder, tissue_probs)
            nt_logits = self.nuclei_type_maps_decoder.decoder0_header(pre_header)
        elif self.tissue_fusion == "cross_attn_bottleneck":
            tissue_probs = torch.softmax(tissue_map, dim=1)
            if self._current_epoch < self.fusion_warmup_epochs:
                tissue_probs = torch.zeros_like(tissue_probs)
            pre_header = self._forward_upsample_features_cross_attn_bn(z0, z1, z2, z3, z4, self.nuclei_type_maps_decoder, tissue_probs)
            nt_logits = self.nuclei_type_maps_decoder.decoder0_header(pre_header)
        else:
            nt_logits = self._forward_upsample(z0, z1, z2, z3, z4, self.nuclei_type_maps_decoder)

        if self.use_compatibility_constraint and self._current_epoch >= self.fusion_warmup_epochs:
            nt_logits = self._apply_compatibility_constraint(nt_logits, tissue_map, out_dict["nuclei_binary_map"])

        out_dict["nuclei_type_map"] = nt_logits

        if retrieve_tokens:
            out_dict["tokens"] = z4

        return out_dict
    

    def _forward_upsample_features(self, z0, z1, z2, z3, z4, branch_decoder):
        b4 = branch_decoder.bottleneck_upsampler(z4)
        b3 = self.decoder3(z3)
        b3 = branch_decoder.decoder3_upsampler(torch.cat([b3, b4], dim=1))
        b2 = self.decoder2(z2)
        b2 = branch_decoder.decoder2_upsampler(torch.cat([b2, b3], dim=1))
        b1 = self.decoder1(z1)
        b1 = branch_decoder.decoder1_upsampler(torch.cat([b1, b2], dim=1))
        b0 = self.decoder0(z0)
        return torch.cat([b0, b1], dim=1)
    

    def _forward_upsample_features_multi_aff(self, z0, z1, z2, z3, z4, branch_decoder, tissue_probs):
        b4 = branch_decoder.bottleneck_upsampler(z4)
        b3 = self.decoder3(z3)
        b3 = branch_decoder.decoder3_upsampler(torch.cat([b3, b4], dim=1))
        b3 = self.multi_aff.aff_b3(b3, tissue_probs)

        b2 = self.decoder2(z2)
        b2 = branch_decoder.decoder2_upsampler(torch.cat([b2, b3], dim=1))
        b2 = self.multi_aff.aff_b2(b2, tissue_probs)

        b1 = self.decoder1(z1)
        b1 = branch_decoder.decoder1_upsampler(torch.cat([b1, b2], dim=1))
        b1 = self.multi_aff.aff_b1(b1, tissue_probs)

        b0 = self.decoder0(z0)
        return torch.cat([b0, b1], dim=1)
    

    def _forward_upsample_features_cross_attn_bn(self, z0, z1, z2, z3, z4, branch_decoder, tissue_probs):
        b4 = branch_decoder.bottleneck_upsampler(z4)
        b4 = self.cross_attn_bn(b4, tissue_probs)
        b3 = self.decoder3(z3)
        b3 = branch_decoder.decoder3_upsampler(torch.cat([b3, b4], dim=1))
        b2 = self.decoder2(z2)
        b2 = branch_decoder.decoder2_upsampler(torch.cat([b2, b3], dim=1))
        b1 = self.decoder1(z1)
        b1 = branch_decoder.decoder1_upsampler(torch.cat([b1, b2], dim=1))
        b0 = self.decoder0(z0)
        return torch.cat([b0, b1], dim=1)
    