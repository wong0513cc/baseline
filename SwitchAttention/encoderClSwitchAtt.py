from typing import Dict, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from switchAttention import SwitchMultiModalBlock, PreNorm, MLP, SwitchEncoder
from loss import Projector, multimodal_icl_monthly

# --------------------------
# Utils
# --------------------------
class SafeLayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=1e5, neginf=-1e5)
        return super().forward(x)

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # [max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        支援：
          - 3D: [B, K, D]（例如 B*N, K, H）
          - 4D: [B, K, N, D]
        """
        K = x.size(1)
        if x.dim() == 3:
            # x: [B, K, D]
            return x + self.pe[:K].to(x.device).unsqueeze(0)       # -> [1, K, D]
        elif x.dim() == 4:
            # x: [B, K, N, D]
            return x + self.pe[:K].to(x.device).unsqueeze(0).unsqueeze(2)  # -> [1, K, 1, D]
        else:
            raise ValueError(f"SinusoidalPositionalEncoding expects 3D or 4D, got {x.dim()}D")
        
class BoundedHead(nn.Module):
    def __init__(self, in_dim: int, lo: float = 0.0, hi: float = 1.0, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            SafeLayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, 1)
        )
        self.register_buffer("lo", torch.tensor(float(lo)))
        self.register_buffer("hi", torch.tensor(float(hi)))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return torch.sigmoid(z) * (self.hi - self.lo) + self.lo

# --------------------------
# Encoders (no attention)
# --------------------------
class LSTMTimeEncoder(nn.Module):
    def __init__(self, d_in: int, hidden: int, num_layers: int = 1, bidirectional: bool = False, dropout: float = 0.1, use_posenc: bool = False):
        super().__init__()
        self.proj_in = nn.Linear(d_in, hidden)
        h = hidden // 2 if bidirectional else hidden
        self.lstm = nn.LSTM(
            input_size=hidden, hidden_size=h, num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        out_dim = h * (2 if bidirectional else 1)
        self.proj_out = nn.Linear(out_dim, hidden) if out_dim != hidden else nn.Identity()
        self.posenc = SinusoidalPositionalEncoding(hidden) if use_posenc else nn.Identity()
        self.norm = SafeLayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, K, N, Din]  ->  out: [B, K, N, H]
        B, K, N, _ = x.shape
        x = self.proj_in(x)                                  # [B,K,N,H]
        seq = x.permute(0, 2, 1, 3).reshape(B * N, K, -1)    # [B*N,K,H]
        seq = self.posenc(seq)
        h, _ = self.lstm(seq)                                # [B*N,K,hidden or 2*hidden/2]
        h = self.proj_out(h)                                 # [B*N,K,H]
        h = self.norm(h)
        return h.reshape(B, N, K, -1).permute(0, 2, 1, 3)    # [B,K,N,H]


class TransformerTimeEncoder(nn.Module):
    def __init__(self, d_in: int, d_model: int, nhead: int = 4, num_layers: int = 2, dim_ff: int = 4, dropout: float = 0.1, use_posenc: bool = True):
        super().__init__()
        self.proj_in = nn.Linear(d_in, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * dim_ff, dropout=dropout,
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.posenc = SinusoidalPositionalEncoding(d_model) if use_posenc else nn.Identity()
        self.norm = SafeLayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, K, N, _ = x.shape
        x = self.proj_in(x)                                   # [B,K,N,H]
        seq = x.permute(0, 2, 1, 3).reshape(B * N, K, -1)     # [B*N,K,H]
        seq = self.posenc(seq)
        h = self.encoder(seq)                                 # [B*N,K,H]
        h = self.norm(h)
        return h.reshape(B, N, K, -1).permute(0, 2, 1, 3)     # [B,K,N,H]
    

# --------------------------
# Information Coefficient (IC) loss
# --------------------------
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

def _rank_transform(v: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Approximate ranks (1..L) along `dim`. For ties we use argsort-of-argsort方式（簡化處理）。
    """
    # permute target dim to last for simpler handling
    perm = list(range(v.dim()))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    invperm = [0]*v.dim()
    for i,p in enumerate(perm): invperm[p] = i
    x = v.permute(*perm).contiguous()  # [..., L]
    idx1 = torch.argsort(x, dim=-1, stable=True)
    ranks = torch.argsort(idx1, dim=-1, stable=True).float() + 1.0
    return ranks.permute(*invperm)

def supcon_loss(z: torch.Tensor, y: torch.Tensor, tau: float = 0.1):
    """
    z: [B, N, D]   融合或單模態的公司嵌入（已時間池化）
    y: [B, N]      對應的「類別」(int, 已離散化)
    """
    B, N, D = z.shape
    z = torch.nn.functional.normalize(z.reshape(B*N, D), dim=-1)  # [BN, D]
    y = y.reshape(B*N)  # [BN]

    # mask: 同類別且非自身
    mask = (y.unsqueeze(0) == y.unsqueeze(1))  # [BN, BN]
    self_mask = torch.eye(B*N, dtype=torch.bool, device=z.device)
    mask = mask & ~self_mask

    # 相似度
    sim = z @ z.T / tau  # [BN, BN]

    # 對每一行計算 log-softmax，並只對正樣本取平均
    sim_max = sim.max(dim=1, keepdim=True).values.detach()  # 數值穩定
    logits = sim - sim_max
    exp_logits = torch.exp(logits) * (~self_mask)  # 排除自己
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    # 正樣本平均
    pos_log_prob = (log_prob * mask).sum(dim=1) / (mask.sum(dim=1).clamp_min(1))
    loss = -pos_log_prob.mean()
    return loss



# --------------------------
# Main model (no cross-attn; concat then MLP)
# --------------------------
class ESGMultiModalModel(nn.Module):
    def __init__(self,
                 d_price: int, 
                 d_finance: int, 
                 d_news: int, 
                 d_event: int,
                 hidden: int = 64,
                 lstm_layers: int = 1, 
                 lstm_bidirectional: bool = False,
                 dropout: float = 0.1,
                 nhead_time: int = 4,
                 news_layers: int = 2, 
                 event_layers: int = 2,
                 ic_weight: float = 0.2,    # 你原本用來記錄 IC(company) 的權重（如果要進 total）
                 ic_type: str = "pearson",
                 icl_weight: float = 0.2,   # NEW: 對比損失的權重
                 icl_tau: float = 0.07      # NEW: 對比損失的溫度
                 ):
        super().__init__()
        self.ic_weight = ic_weight
        self.ic_type = ic_type
        self.icl_weight = icl_weight        # NEW
        self.icl_tau = icl_tau              # NEW

        # encoders（略，沿用你的）
        self.enc_price = LSTMTimeEncoder(d_in=d_price, hidden=hidden,
                                         num_layers=lstm_layers, bidirectional=lstm_bidirectional,
                                         dropout=dropout, use_posenc=False)
        self.enc_fin   = LSTMTimeEncoder(d_in=d_finance, hidden=hidden,
                                         num_layers=lstm_layers, bidirectional=lstm_bidirectional,
                                         dropout=dropout, use_posenc=False)
        self.enc_news  = TransformerTimeEncoder(d_in=d_news,  d_model=hidden,
                                                nhead=nhead_time, num_layers=news_layers, dropout=dropout)
        self.enc_event = TransformerTimeEncoder(d_in=d_event, d_model=hidden,
                                                nhead=nhead_time, num_layers=event_layers, dropout=dropout)

        self.switch = SwitchEncoder(hidden_dim=hidden, depth=6, num_heads=4, dropout=dropout)

        d_fused = hidden * 4
        self.company_head = BoundedHead(in_dim=d_fused, lo=0.0, hi=1.0, dropout=dropout)

        # NEW: 四個 projector（只用來算 ICL，不走預測頭）
        self.projectors = nn.ModuleDict({
            'price':  Projector(hidden),
            'finance':Projector(hidden),
            'news':   Projector(hidden),
            'event':  Projector(hidden),
        })

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        price, finance, news, event = batch["price"], batch["finance"], batch["news"], batch["event"]  # 注意你上方 concat 用的是 E,Nw 的順序；這裡保持一致
        label_company = batch.get("label_company", None)  # [B,1,N,1]
        valid_mask_dict = batch.get("valid_mask_dict", None)  # 可選：{'price':[B,K,N] bool, ...}

        # 1) encoders -> [B,K,N,H]
        Hp = self.enc_price(price)
        Hf = self.enc_fin(finance)
        Hn = self.enc_news(news)
        He = self.enc_event(event)

        # 2) cross-modal（沿用你的 switch）
        updated_list, fused = self.switch([Hp, Hf, Hn, He])

        # 3) 時間池化 -> [B,N,H]
        P  = updated_list[0].mean(dim=1)
        F  = updated_list[1].mean(dim=1)
        Nw = updated_list[2].mean(dim=1)
        E  = updated_list[3].mean(dim=1)

        # 4) 融合 + 預測
        Z = torch.cat([P, F, E, Nw], dim=-1)   # 與你上方一致：P,F,E,Nw
        pred_company = self.company_head(Z).unsqueeze(1)  # [B,1,N,1]

        out = {"pred_company": pred_company}

        # ====== NEW: ICL（六配對，逐月平均；公司維當 batch） ======
        # 用「encoder 輸出」來算對比（預測路徑仍不吃 projector）
        H_dict = {'price': Hp, 'finance': Hf, 'news': Hn, 'event': He}
        loss_icl = multimodal_icl_monthly(
            H_dict=H_dict,
            projector_dict=self.projectors,
            valid_mask_dict=valid_mask_dict,
            tau=self.icl_tau
        )
        # ===========================================

        # 5) 監督式損失（公司層級）
        if label_company is not None:
            p = pred_company                  # [B,1,N,1]
            t = label_company                 # [B,1,N,1]
            valid = ~torch.isnan(t)
            diff2 = (torch.nan_to_num(p - t, nan=0.0) ** 2) * valid.float()
            denom = valid.float().sum().clamp_min(1.0)
            mse_company = diff2.sum() / denom

            pN = torch.nan_to_num(p.squeeze(1).squeeze(-1), nan=0.0)  # [B,N]
            tN = torch.nan_to_num(t.squeeze(1).squeeze(-1), nan=0.0)  # [B,N]
            if self.ic_type.lower().startswith("pear"):
                ic_per_b = _pearson_corr(pN, tN, dim=1)               # [B]
            else:
                rp = _rank_transform(pN, dim=1)
                rt = _rank_transform(tN, dim=1)
                ic_per_b = _pearson_corr(rp, rt, dim=1)
            ic_company = ic_per_b.mean()

            # NEW: 把 ICL 納入 total（IC 指標先不進 total；若想進，自己加 -self.ic_weight*ic_company）
            total = mse_company + self.icl_weight * loss_icl

            out["losses"] = {
                "mse": mse_company,
                "ic_company": ic_company,  # 指標用；非 loss
                "icl": loss_icl,           # NEW: 對比損失
                "total": total
            }

        return out





