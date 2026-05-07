import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-3, ignore_classes=None):
        super().__init__()
        self.smooth = smooth
        self.ignore_classes = set(ignore_classes) if ignore_classes else set()

    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W]
            target: [B, H, W]
        """
        num_classes = pred.size(1)
        pred = F.softmax(pred, dim=1)

        if num_classes == 2:
            pred_pos = pred[:, 1, :, :].contiguous().view(-1)
            target_bin = target.float().contiguous().view(-1)

            intersection = (pred_pos * target_bin).sum()
            dice = (2. * intersection + self.smooth) / (pred_pos.sum() + target_bin.sum() + self.smooth)
            return 1 - dice
        else:
            target_one_hot = F.one_hot(target.clamp(0, num_classes - 1), num_classes=num_classes)  # [B, H, W, C]
            target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

            if self.ignore_classes:
                valid_mask = torch.ones(target.shape, dtype=torch.bool, device=target.device)
                for ig_c in self.ignore_classes:
                    valid_mask &= (target != ig_c)
                valid_flat = valid_mask.contiguous().view(-1)
            else:
                valid_flat = None

            dice_sum = 0.0
            valid_classes = 0
            for c in range(1, num_classes):
                pred_c = pred[:, c].contiguous().view(-1)
                target_c = target_one_hot[:, c].contiguous().view(-1)

                if valid_flat is not None:
                    pred_c = pred_c[valid_flat]
                    target_c = target_c[valid_flat]

                if target_c.sum() == 0:
                    continue

                intersection = (pred_c * target_c).sum()
                dice_c = (2. * intersection + self.smooth) / (pred_c.sum() + target_c.sum() + self.smooth)
                dice_sum += dice_c
                valid_classes += 1

            if valid_classes == 0:
                return torch.tensor(0.0, device=pred.device)
            return 1 - dice_sum / valid_classes


class MSGELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def get_sobel_kernel(self, size):
        assert size % 2 == 1, "size must be odd"

        h_range = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32)
        v_range = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32)

        h, v = torch.meshgrid(h_range, v_range, indexing='ij')

        kernel_h = h / (h * h + v * v + 1e-15)
        kernel_v = v / (h * h + v * v + 1e-15)

        return kernel_h, kernel_v

    def get_gradient(self, x, kernel_h, kernel_v):
        b, c, h, w = x.shape

        kernel_h = kernel_h.view(1, 1, kernel_h.size(0), kernel_h.size(1)).to(x.device)
        kernel_v = kernel_v.view(1, 1, kernel_v.size(0), kernel_v.size(1)).to(x.device)

        kernel_h = kernel_h.repeat(c, 1, 1, 1)
        kernel_v = kernel_v.repeat(c, 1, 1, 1)

        grad_h = F.conv2d(x, kernel_h, padding=kernel_h.size(2)//2, groups=c)
        grad_v = F.conv2d(x, kernel_v, padding=kernel_v.size(2)//2, groups=c)

        return grad_h, grad_v

    def forward(self, pred, target, focus=None):
        """
        Args:
            pred: [B, 2, H, W] 
            target: [B, 2, H, W] 
            focus: [B, H, W] or [B, 1, H, W] - binary nuclei mask 
        """
        kernel_h, kernel_v = self.get_sobel_kernel(5)

        pred_grad_h, pred_grad_v = self.get_gradient(pred, kernel_h, kernel_v)
        target_grad_h, target_grad_v = self.get_gradient(target, kernel_h, kernel_v)

        if focus is not None:
            if focus.dim() == 3:
                focus = focus.unsqueeze(1)  # [B, 1, H, W]
            focus = focus.float()
            focus = focus.expand_as(pred_grad_h)  # [B, 2, H, W]
            diff_h = (pred_grad_h - target_grad_h) ** 2 * focus
            diff_v = (pred_grad_v - target_grad_v) ** 2 * focus
            num_pixels = focus.sum().clamp(min=1.0)
            loss_h = diff_h.sum() / num_pixels
            loss_v = diff_v.sum() / num_pixels
        else:
            loss_h = F.mse_loss(pred_grad_h, target_grad_h)
            loss_v = F.mse_loss(pred_grad_v, target_grad_v)

        return loss_h + loss_v


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None, ignore_index: int = -100, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        if self.weight is not None and self.weight.device != logits.device:
            self.weight = self.weight.to(logits.device)
        ce = F.cross_entropy(logits, target, weight=self.weight, ignore_index=self.ignore_index, 
                             label_smoothing=self.label_smoothing, reduction='none')
        with torch.no_grad():
            valid = (target != self.ignore_index)
            target_safe = target.clamp(min=0)
        logp = F.log_softmax(logits, dim=1)
        p_t = logp.gather(1, target_safe.unsqueeze(1)).squeeze(1).exp()
        focal_w = (1.0 - p_t).clamp(min=0.0) ** self.gamma
        loss = focal_w * ce
        if valid.any():
            return loss[valid].mean()
        return loss.sum() * 0.0


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=4.0/3.0, smooth=1e-6, ignore_classes=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.ignore_classes = set(ignore_classes) if ignore_classes else set()

    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W]
            target: [B, H, W]
        """
        num_classes = pred.size(1)
        pred = F.softmax(pred, dim=1)

        target_one_hot = F.one_hot(target.clamp(0, num_classes - 1), num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        pred_flat = pred.view(pred.size(0), pred.size(1), -1)   # [B, C, H*W]
        target_flat = target_one_hot.view(target_one_hot.size(0), target_one_hot.size(1), -1)

        if self.ignore_classes:
            valid_mask = torch.ones(target.shape, dtype=torch.bool, device=target.device)
            for ig_c in self.ignore_classes:
                valid_mask &= (target != ig_c)
            mask_flat = valid_mask.view(valid_mask.size(0), 1, -1).float()  # [B, 1, H*W]
            pred_flat = pred_flat * mask_flat
            target_flat = target_flat * mask_flat

            keep = [c for c in range(num_classes) if c not in self.ignore_classes]
            pred_flat = pred_flat[:, keep, :]
            target_flat = target_flat[:, keep, :]

        TP = (pred_flat * target_flat).sum(dim=2)
        FP = ((1 - target_flat) * pred_flat).sum(dim=2)
        FN = (target_flat * (1 - pred_flat)).sum(dim=2)

        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        focal_tversky = (1 - tversky_index).clamp(min=0.0) ** self.gamma

        return focal_tversky.mean()


class CellViTMultiTaskLoss(nn.Module):
    def __init__(
        self,
        lambda_np_ft: float = 1.0,
        lambda_np_dice: float = 1.0,
        lambda_hv_mse: float = 2.5,
        lambda_hv_msge: float = 8.0,
        lambda_nt_ft: float = 0.5,
        lambda_nt_dice: float = 0.2,
        lambda_nt_bce: float = 0.5,
        lambda_tc_ce: float = 0.1,
        ft_alpha: float = 0.7,
        ft_beta: float = 0.3,
        ft_gamma: float = 4.0 / 3.0,
        ft_eps: float = 1e-6,
        unlabeled_class: int = None,
        nt_class_weights: list = None,
    ):
        super().__init__()
        self.lambda_np_ft = lambda_np_ft
        self.lambda_np_dice = lambda_np_dice
        self.lambda_hv_mse = lambda_hv_mse
        self.lambda_hv_msge = lambda_hv_msge
        self.lambda_nt_ft = lambda_nt_ft
        self.lambda_nt_dice = lambda_nt_dice
        self.lambda_nt_bce = lambda_nt_bce
        self.lambda_tc_ce = lambda_tc_ce

        ignore = [unlabeled_class] if unlabeled_class is not None else []
        ignore_index = unlabeled_class if unlabeled_class is not None else -100

        # np branch
        self.focal_tversky_np = FocalTverskyLoss(alpha=ft_alpha, beta=ft_beta, gamma=ft_gamma, smooth=ft_eps)
        self.dice_np = DiceLoss()

        # hv branch
        self.mse = nn.MSELoss()
        self.msge = MSGELoss()

        # nt branch
        self.focal_tversky_nt = FocalTverskyLoss(alpha=ft_alpha, beta=ft_beta, gamma=ft_gamma, 
                                                 smooth=ft_eps, ignore_classes=ignore)
        self.dice_nt = DiceLoss(ignore_classes=ignore)
        bce_weight = torch.tensor(nt_class_weights, dtype=torch.float32) if nt_class_weights is not None else None
        self.bce_nt = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=bce_weight)

        # tissue branch
        self.ce_tc = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        loss_dict = {}

        out_bin  = outputs["nuclei_binary_map"].float()
        out_hv   = outputs["hv_map"].float()
        out_type = outputs["nuclei_type_map"].float()
        tgt_hv   = targets["hv_map"].float()
        tgt_bin  = targets["nuclei_binary_map"]
        tgt_type = targets["nuclei_type_map"]

        np_ft = self.focal_tversky_np(out_bin, tgt_bin)
        np_dice = self.dice_np(out_bin, tgt_bin)
        l_np = self.lambda_np_ft * np_ft + self.lambda_np_dice * np_dice
        loss_dict['np_ft'] = np_ft.item()
        loss_dict['np_dice'] = np_dice.item()
        loss_dict['np_loss'] = l_np.item()

        hv_mse = self.mse(out_hv, tgt_hv)
        hv_msge = self.msge(out_hv, tgt_hv, focus=tgt_bin)
        l_hv = self.lambda_hv_mse * hv_mse + self.lambda_hv_msge * hv_msge
        loss_dict['hv_mse'] = hv_mse.item()
        loss_dict['hv_msge'] = hv_msge.item()
        loss_dict['hv_loss'] = l_hv.item()

        if self.bce_nt.weight is not None:
            self.bce_nt.weight = self.bce_nt.weight.to(out_type.device)
        nt_ft = self.focal_tversky_nt(out_type, tgt_type)
        nt_dice = self.dice_nt(out_type, tgt_type)
        _nt_valid = (tgt_type != self.bce_nt.ignore_index).any()
        nt_bce = self.bce_nt(out_type, tgt_type) if _nt_valid else torch.tensor(0.0, device=out_type.device)
        l_nt = self.lambda_nt_ft * nt_ft + self.lambda_nt_dice * nt_dice + self.lambda_nt_bce * nt_bce
        loss_dict['nt_ft'] = nt_ft.item()
        loss_dict['nt_dice'] = nt_dice.item()
        loss_dict['nt_bce'] = nt_bce.item()
        loss_dict['nt_loss'] = l_nt.item()

        l_tc = torch.tensor(0.0, device=out_bin.device)
        if self.lambda_tc_ce > 0 and "tissue_types" in outputs and "tissue_mask" in targets:
            if targets["tissue_mask"] is not None:
                tissue_mask = targets["tissue_mask"]  # [B, H, W]
                tissue_label = tissue_mask.view(tissue_mask.size(0), -1).mode(dim=1).values  # [B,]
                tc_ce = self.ce_tc(outputs["tissue_types"].float(), tissue_label)
                l_tc = self.lambda_tc_ce * tc_ce
                loss_dict['tc_loss'] = l_tc.item()

        total_loss = l_np + l_hv + l_nt + l_tc

        return total_loss, loss_dict


class CellViTTissueLoss(CellViTMultiTaskLoss):
    def __init__(
        self,
        lambda_ts_ft: float = 1.0,
        lambda_ts_dice: float = 0.5,
        lambda_ts_ce: float = 1.0,
        tissue_ignore_classes: list = None,
        ts_class_weights: list = None,
        tissue_label_smoothing: float = 0.0,
        tissue_dedup: bool = True,
        tissue_use_focal: bool = False,
        tissue_focal_gamma: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lambda_ts_ft = lambda_ts_ft
        self.lambda_ts_dice = lambda_ts_dice
        self.lambda_ts_ce = lambda_ts_ce
        self.tissue_dedup = tissue_dedup

        tissue_ignore = tissue_ignore_classes if tissue_ignore_classes else [0]
        self._tissue_ignore_set = set(tissue_ignore)
        self._ce_ignore_val = tissue_ignore[0]

        ft_alpha = kwargs.get("ft_alpha", 0.7)
        ft_beta = kwargs.get("ft_beta", 0.3)
        ft_gamma = kwargs.get("ft_gamma", 4.0 / 3.0)
        ft_eps = kwargs.get("ft_eps", 1e-6)

        self.focal_tversky_ts = FocalTverskyLoss(alpha=ft_alpha, beta=ft_beta, gamma=ft_gamma, smooth=ft_eps,
                                                 ignore_classes=tissue_ignore)
        self.dice_ts = DiceLoss(ignore_classes=tissue_ignore)
        ce_weight = torch.tensor(ts_class_weights, dtype=torch.float32) if ts_class_weights else None
        if tissue_use_focal:
            self.ce_ts = FocalLoss(gamma=tissue_focal_gamma, weight=ce_weight, ignore_index=self._ce_ignore_val,
                                   label_smoothing=tissue_label_smoothing)
        else:
            self.ce_ts = nn.CrossEntropyLoss(ignore_index=self._ce_ignore_val, weight=ce_weight,
                                             label_smoothing=tissue_label_smoothing)

    def forward(self, outputs, targets):
        total_loss, loss_dict = super().forward(outputs, targets)

        l_ts = torch.tensor(0.0, device=total_loss.device)
        active = self.lambda_ts_ft + self.lambda_ts_dice + self.lambda_ts_ce
        if active > 0 and "tissue_segmentation_map" in outputs and "tissue_mask" in targets:
            out_ts = outputs.get("tissue_segmentation_map_full", outputs["tissue_segmentation_map"]).float()
            tgt_ts = targets.get("tissue_mask_context", targets["tissue_mask"])  # (B, H, W)

            patch_idx = targets.get("patch_idx")
            if self.tissue_dedup and patch_idx is not None:
                first_mask = (patch_idx == 0)
                if first_mask.any():
                    out_ts = out_ts[first_mask]
                    tgt_ts = tgt_ts[first_mask]
                else:
                    loss_dict["ts_ft"] = 0.0
                    loss_dict["ts_dice"] = 0.0
                    loss_dict["ts_ce"] = 0.0
                    loss_dict["ts_loss"] = 0.0
                    return total_loss + l_ts, loss_dict

            if self.ce_ts.weight is not None:
                self.ce_ts.weight = self.ce_ts.weight.to(out_ts.device)

            if tgt_ts.shape[-2:] != out_ts.shape[-2:]:
                tgt_ts = F.interpolate(tgt_ts.unsqueeze(1).float(), size=out_ts.shape[-2:], 
                                       mode='nearest').squeeze(1).long()

            ts_ft   = self.focal_tversky_ts(out_ts, tgt_ts)
            ts_dice = self.dice_ts(out_ts, tgt_ts)

            tgt_ce = tgt_ts
            if len(self._tissue_ignore_set) > 1:
                tgt_ce = tgt_ts.clone()
                for c in self._tissue_ignore_set:
                    if c != self._ce_ignore_val:
                        tgt_ce[tgt_ts == c] = self._ce_ignore_val
            _ts_valid = (tgt_ce != self._ce_ignore_val).any()
            ts_ce = self.ce_ts(out_ts, tgt_ce) if _ts_valid else torch.tensor(0.0, device=out_ts.device)

            l_ts = (self.lambda_ts_ft * ts_ft + self.lambda_ts_dice * ts_dice + self.lambda_ts_ce * ts_ce)
            loss_dict["ts_ft"]   = ts_ft.item()
            loss_dict["ts_dice"] = ts_dice.item()
            loss_dict["ts_ce"]   = ts_ce.item()
            loss_dict["ts_loss"] = l_ts.item()

        return total_loss + l_ts, loss_dict
