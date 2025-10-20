import torch
import torch.nn.functional as F

def supcon_loss(z: torch.Tensor, y: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """
    z: [B, N, D]  公司嵌入（不要 detach）
    y: [B, N]     類別 id（int），已離散化（例如 0..C-1）
    """
    B, N, D = z.shape
    z = F.normalize(z.reshape(B*N, D), dim=-1)  # [BN, D]
    y = y.reshape(B*N)                          # [BN]

    # 同類別且非自身的 mask
    mask = (y.unsqueeze(0) == y.unsqueeze(1))   # [BN, BN]
    self_mask = torch.eye(B*N, dtype=torch.bool, device=z.device)
    mask = mask & ~self_mask

    # 相似度 / 溫度
    sim = (z @ z.T) / tau                       # [BN, BN]

    # 數值穩定 + 排除自己
    sim_max = sim.max(dim=1, keepdim=True).values.detach()
    logits = sim - sim_max
    exp_logits = torch.exp(logits) * (~self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    # 只對正樣本取平均；若某行沒有正樣本 → 分母設為 1 避免除 0
    pos_cnt = mask.sum(dim=1).clamp_min(1)
    pos_log_prob = (log_prob * mask).sum(dim=1) / pos_cnt
    loss = -pos_log_prob.mean()
    return loss
