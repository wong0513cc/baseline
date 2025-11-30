import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

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




def _pairwise_distance(x: torch.Tensor,
                       y: torch.Tensor,
                       distance: str = "cosine") -> torch.Tensor:
    """
    x: [M, D]
    y: [M, D]
    回傳 dist: [M, M]，dist[i,j] = d(x_i, y_j)
    """
    if distance == "cosine":
        x_norm = F.normalize(x, p=2, dim=-1)
        y_norm = F.normalize(y, p=2, dim=-1)
        sim = x_norm @ y_norm.t()       # [M,M]
        dist = 1.0 - sim                # cosine distance
        return dist.clamp_min(0.0)
    elif distance == "euclidean":
        # ||x_i - y_j||_2
        diff = x.unsqueeze(1) - y.unsqueeze(0)   # [M,1,D] - [1,M,D] -> [M,M,D]
        dist = diff.pow(2).sum(-1).sqrt()        # [M,M]
        return dist
    else:
        raise ValueError(f"Unknown distance type: {distance}")


def _triplet_loss_batch_all(
    emb_a: torch.Tensor,     # [M,D] anchor modality
    emb_b: torch.Tensor,     # [M,D] other modality (same or different)
    labels_a: torch.Tensor,  # [M]
    labels_b: torch.Tensor,  # [M]
    margin: float = 0.2,
    distance: str = "cosine",
    exclude_self: bool = False,
) -> torch.Tensor:
    """
    多模態 triplet loss 的一個 term（例如 L_PP, L_PF 等）
    - emb_a: anchor 的 embedding（P/F/N/E 其中一個）
    - emb_b: positive/negative 的 embedding（可以同模態或不同模態）
    - labels_*: 對應的 class / 身份 label
    - exclude_self: within-modal 時要排除 (i,i) 這種 trivial pair

    策略：batch-all（對每個 anchor i，和所有 positive/negative 做組合）
    計算量大約 O(M^2)，對 M ~ 1000 還可以接受。
    """
    device = emb_a.device
    labels_a = labels_a.view(-1).to(device)
    labels_b = labels_b.view(-1).to(device)

    M, D = emb_a.shape
    assert emb_b.shape == (M, D), f"emb_b shape {emb_b.shape} != {(M, D)}"

    # [M,M]
    dist = _pairwise_distance(emb_a, emb_b, distance=distance)

    # 正負樣本 mask
    labels_equal = labels_a.unsqueeze(1) == labels_b.unsqueeze(0)   # [M,M]
    pos_mask = labels_equal.clone()
    neg_mask = ~labels_equal

    if exclude_self:
        # within-modal: 把 (i,i) 從 positive 裡移除
        eye = torch.eye(M, dtype=torch.bool, device=device)
        pos_mask = pos_mask & (~eye)

    # 每個 anchor 至少要有一個 positive 和 negative 才算
    pos_any = pos_mask.any(dim=1)   # [M]
    neg_any = neg_mask.any(dim=1)   # [M]
    valid_anchor = pos_any & neg_any

    if not valid_anchor.any():
        # 沒有 valid triplet，就回傳 0（不影響其他 term）
        return dist.new_tensor(0.0)

    total_loss = dist.new_tensor(0.0)
    total_triplets = dist.new_tensor(0.0)

    # 對每個 anchor i 個別處理，避免建 MxMxM 的超大 tensor
    for i in torch.nonzero(valid_anchor, as_tuple=False).view(-1):
        pos_idx = torch.nonzero(pos_mask[i], as_tuple=False).view(-1)
        neg_idx = torch.nonzero(neg_mask[i], as_tuple=False).view(-1)

        # shape: [P]、[N]
        d_ap = dist[i, pos_idx].unsqueeze(1)   # [P,1]
        d_an = dist[i, neg_idx].unsqueeze(0)   # [1,N]

        # margin + d(ap) - d(an)
        loss_ij = margin + d_ap - d_an         # [P,N]
        loss_ij = F.relu(loss_ij)

        total_loss += loss_ij.sum()
        total_triplets += (loss_ij > 0).float().sum()

    if total_triplets < 1.0:
        # 幾乎沒 valid triplet，就平均全部
        return total_loss / (M + 1e-6)
    else:
        return total_loss / total_triplets
    
def multimodal_triplet_loss_4modal(
        embeds: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        margin_within: float = 0.2,
        margin_cross: float = 0.2,
        gamma_within: Dict[str, float] = None,
        gamma_cross: Dict[Tuple[str, str], float] = None,
        distance: str = "cosine",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        四模態 + 10 個 triplet loss term（4 個 within + 6 個 cross）

        embeds: dict，key 必須包含：
            "price", "fin", "news", "event"
            每個 tensor shape: [B,N,D] or [B,N,D,...]（後面會 flatten 成 [M,D]）

        labels: [B,N] 或 [M] 的 class / 身份（int），例如公司 ID 或離散化 label

        margin_within: 同模態 triplet 的 margin（α1, α2,...）
        margin_cross:  跨模態 triplet 的 margin（α3 類）

        gamma_within: dict，控制每個 within term 的 γ，例如：
            {"price":1.0, "fin":1.0, "news":1.0, "event":1.0}
            如果是 None，就全部設為 1.0

        gamma_cross: dict，控制每個 cross pair 的 γ，例如：
            {("price","fin"):1.0, ("price","news"):1.0, ...}
            如果是 None，就全部設為 1.0

        distance: "cosine" or "euclidean"

        回傳：
            total_loss: scalar
            loss_dict: 例如 {"PP":..., "FF":..., "PF":..., ...}
        """
        # 準備 embedding：flatten 成 [M,D]
        z_p = embeds["price"].reshape(-1, embeds["price"].shape[-1])
        z_f = embeds["fin"].reshape(-1, embeds["fin"].shape[-1])
        z_n = embeds["news"].reshape(-1, embeds["news"].shape[-1])
        z_e = embeds["event"].reshape(-1, embeds["event"].shape[-1])

        labels_flat = labels.reshape(-1)

        device = z_p.device
        labels_flat = labels_flat.to(device)

        # 預設權重
        if gamma_within is None:
            gamma_within = {
                "price": 1.0,
                "fin": 1.0,
                "news": 1.0,
                "event": 1.0,
            }

        if gamma_cross is None:
            gamma_cross = {
                ("price", "fin"): 1.0,
                ("price", "news"): 1.0,
                ("price", "event"): 1.0,
                ("fin", "news"): 1.0,
                ("fin", "event"): 1.0,
                ("news", "event"): 1.0,
            }

        loss_dict = {}

        # 1) within-modal：PP, FF, NN, EE
        L_PP = _triplet_loss_batch_all(z_p, z_p, labels_flat, labels_flat,
                                    margin=margin_within, distance=distance,
                                    exclude_self=True)
        L_FF = _triplet_loss_batch_all(z_f, z_f, labels_flat, labels_flat,
                                    margin=margin_within, distance=distance,
                                    exclude_self=True)
        L_NN = _triplet_loss_batch_all(z_n, z_n, labels_flat, labels_flat,
                                    margin=margin_within, distance=distance,
                                    exclude_self=True)
        L_EE = _triplet_loss_batch_all(z_e, z_e, labels_flat, labels_flat,
                                    margin=margin_within, distance=distance,
                                    exclude_self=True)

        loss_dict["PP"] = L_PP
        loss_dict["FF"] = L_FF
        loss_dict["NN"] = L_NN
        loss_dict["EE"] = L_EE

        # 2) cross-modal：PF, PN, PE, FN, FE, NE
        L_PF = _triplet_loss_batch_all(z_p, z_f, labels_flat, labels_flat,
                                    margin=margin_cross, distance=distance,
                                    exclude_self=False)
        L_PN = _triplet_loss_batch_all(z_p, z_n, labels_flat, labels_flat,
                                    margin=margin_cross, distance=distance,
                                    exclude_self=False)
        L_PE = _triplet_loss_batch_all(z_p, z_e, labels_flat, labels_flat,
                                    margin=margin_cross, distance=distance,
                                    exclude_self=False)
        L_FN = _triplet_loss_batch_all(z_f, z_n, labels_flat, labels_flat,
                                    margin=margin_cross, distance=distance,
                                    exclude_self=False)
        L_FE = _triplet_loss_batch_all(z_f, z_e, labels_flat, labels_flat,
                                    margin=margin_cross, distance=distance,
                                    exclude_self=False)
        L_NE = _triplet_loss_batch_all(z_n, z_e, labels_flat, labels_flat,
                                    margin=margin_cross, distance=distance,
                                    exclude_self=False)

        loss_dict["PF"] = L_PF
        loss_dict["PN"] = L_PN
        loss_dict["PE"] = L_PE
        loss_dict["FN"] = L_FN
        loss_dict["FE"] = L_FE
        loss_dict["NE"] = L_NE

        # 3) 總和 + 權重
        total = (
            gamma_within.get("price", 0.0) * L_PP +
            gamma_within.get("fin",   0.0) * L_FF +
            gamma_within.get("news",  0.0) * L_NN +
            gamma_within.get("event", 0.0) * L_EE +
            gamma_cross.get(("price", "fin"),   0.0) * L_PF +
            gamma_cross.get(("price", "news"),  0.0) * L_PN +
            gamma_cross.get(("price", "event"), 0.0) * L_PE +
            gamma_cross.get(("fin", "news"),    0.0) * L_FN +
            gamma_cross.get(("fin", "event"),   0.0) * L_FE +
            gamma_cross.get(("news", "event"),  0.0) * L_NE
        )

        return total, loss_dict