import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# IC loss
def _pearson_corr(x: torch.Tensor, y: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    Pearson correlation along `dim`.
    x,y: same shape; returns correlation with that dim reduced.
    """
    x = torch.nan_to_num(x, nan=0.0)
    y = torch.nan_to_num(y, nan=0.0)
    x_mean = x.mean(dim=dim, keepdim=True)
    y_mean = y.mean(dim=dim, keepdim=True)
    xc = x - x_mean
    yc = y - y_mean
    num = (xc * yc).sum(dim=dim)
    den = torch.sqrt((xc.pow(2).sum(dim=dim) + eps) * (yc.pow(2).sum(dim=dim) + eps))
    return num / den

def l2_normalize(x, dim=-1, eps=1e-8):
    return x / (x.norm(p=2, dim=dim, keepdim=True).clamp(min=eps))

def info_nce_two_modal(
    ZA: torch.Tensor,   # [B,N,H]
    ZB: torch.Tensor,   # [B,N,H]
    tau: float = 0.07,
    maskA: torch.Tensor = None,
    maskB: torch.Tensor = None,
) -> torch.Tensor:
    B, N, H = ZA.shape
    ZA = F.normalize(ZA, dim=-1)
    ZB = F.normalize(ZB, dim=-1)

    M = B * N
    ZA_flat = ZA.reshape(M, H)
    ZB_flat = ZB.reshape(M, H)

    if maskA is not None:
        if maskA.dim() == 3:
            maskA = maskA.squeeze(-1)
        maskA = maskA.reshape(M) > 0
    else:
        maskA = torch.ones(M, dtype=torch.bool, device=ZA.device)

    if maskB is not None:
        if maskB.dim() == 3:
            maskB = maskB.squeeze(-1)
        maskB = maskB.reshape(M) > 0
    else:
        maskB = torch.ones(M, dtype=torch.bool, device=ZA.device)

    valid = maskA & maskB
    if valid.sum() <= 1:
        return ZA.new_tensor(0.0)

    ZA_flat = ZA_flat[valid]
    ZB_flat = ZB_flat[valid]
    M_valid = ZA_flat.size(0)

    logits = torch.matmul(ZA_flat, ZB_flat.t()) / tau  # [M_valid,M_valid]
    labels = torch.arange(M_valid, device=ZA.device)
    loss = F.cross_entropy(logits, labels)
    return loss

def multimodal_info_loss_all_pairs(
    Zp: torch.Tensor,   # [B,N,H]
    Zf: torch.Tensor,   # [B,N,H]
    Zn: torch.Tensor,   # [B,N,H]
    Ze: torch.Tensor,   # [B,N,H]
    tau: float = 0.07,
    mask_price: torch.Tensor = None,
    mask_fin: torch.Tensor = None,
    mask_news: torch.Tensor = None,
    mask_event: torch.Tensor = None,
) -> torch.Tensor:
    Zs   = [Zp, Zf, Zn, Ze]
    Ms   = [mask_price, mask_fin, mask_news, mask_event]

    losses = []
    num_modal = len(Zs)

    for i in range(num_modal):
        for j in range(i + 1, num_modal):
            ZA, ZB = Zs[i], Zs[j]
            if ZA is None or ZB is None:
                continue
            mA, mB = Ms[i], Ms[j]
            loss_ij = info_nce_two_modal(ZA, ZB, tau=tau, maskA=mA, maskB=mB)
            losses.append(loss_ij)

    if not losses:
        return Zp.new_tensor(0.0)

    return sum(losses) / len(losses)