from typing import Dict, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from switchAttention import SwitchMultiModalBlock, PreNorm, MLP, SwitchEncoder

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

def information_coefficient_loss(pred: torch.Tensor, target: torch.Tensor, mode: str = "pearson") -> torch.Tensor:
    """
    Compute IC loss (want to *maximize* correlation, so loss = -IC).
    Here we compute per batch item across time K, then average:
      pred, target: [B, K, 1]
    """
    assert pred.shape == target.shape
    # reduce last dim
    pred = pred.squeeze(-1)   # [B,K]
    target = target.squeeze(-1)  # [B,K]
    if mode.lower().startswith("pear"):
        ic = _pearson_corr(pred, target, dim=1)     # [B]
    elif mode.lower().startswith("spear"):
        rp = _rank_transform(pred, dim=1)
        rt = _rank_transform(target, dim=1)
        ic = _pearson_corr(rp, rt, dim=1)           # [B]
    else:
        raise ValueError("ic mode must be 'pearson' or 'spearman'")
    return -ic.mean()  # loss
6                                                                                                                                                                                                                                                          

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
                 ic_weight: float = 0.2, 
                 ic_type: str = "pearson"):
        super().__init__()
        self.ic_weight = ic_weight
        self.ic_type = ic_type

        # unimodal encoders (LSTM no attention; Transformer encoder)
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

        # Heads
        d_fused = hidden * 4  # concat 四模態
        self.company_head = BoundedHead(in_dim=d_fused, lo=0.0, hi=1.0, dropout=dropout)  # 若標籤是 0~100，hi=100.0
        self.head         = BoundedHead(in_dim=d_fused, lo=0.0, hi=1.0, dropout=dropout)

        # self.head = nn.Sequential(
        #     nn.Linear(d_fused, d_fused),
        #     nn.ReLU(),
        #     SafeLayerNorm(d_fused),
        #     nn.Dropout(dropout),
        #     nn.Linear(d_fused, 1)
        # )


    @torch.no_grad()
    def encode_modalities(self, batch: Dict[str, torch.Tensor], pool: str = "time"):
        """
        只做 encoder，不做後續 fusion/head。
        回傳：
          - raw:  每個模態的 encoder 輸出 [B,K,N,H]
          - pooled: 經過時間池化後的 [B,N,H]（用於可視化）
        pool: "time" -> 平均; 也可以改成 "last"/"attn" 視你的 encoder 設計
        """
        self.eval()
        price = batch["price"]   # [B,K,N,Dp]
        finance = batch["finance"]
        news = batch["news"]
        event = batch["event"]

        # 跑各自的 encoder，這些方法已在你的模型裡
        Hp = self.enc_price(price)     # [B,K,N,H]
        Hf = self.enc_fin(finance) # [B,K,N,H]
        Hn = self.enc_news(news)       # [B,K,N,H]
        He = self.enc_event(event)     # [B,K,N,H]

        if pool == "time":
            Hp_pool = Hp.mean(dim=1)  # [B,N,H]
            Hf_pool = Hf.mean(dim=1)
            Hn_pool = Hn.mean(dim=1)
            He_pool = He.mean(dim=1)
        elif pool == "last":
            Hp_pool = Hp[:, -1]
            Hf_pool = Hf[:, -1]
            Hn_pool = Hn[:, -1]
            He_pool = He[:, -1]
        else:
            raise ValueError(f"Unknown pool={pool}")

        out = {
            "raw":   {"price": Hp, "finance": Hf, "news": Hn, "event": He},
            "pooled":{"price": Hp_pool, "finance": Hf_pool, "news": Hn_pool, "event": He_pool},
        }
        return out

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        price, finance, news, event = batch["price"], batch["finance"], batch["news"], batch["event"]
        label_company = batch.get("label_company", None)  # [B,1,N,1]

        # 1) encoders -> [B,K,N,H]
        Hp = self.enc_price(price)
        Hf = self.enc_fin(finance)
        Hn = self.enc_news(news)
        He = self.enc_event(event)

        # 2) 時間平均（K→1），得到逐公司年度表示 [B,N,H]
        P = Hp.mean(dim=1)
        Fv = Hf.mean(dim=1)
        Nw = Hn.mean(dim=1)
        Ev = He.mean(dim=1)

        # 3) 模態融合：concat → [B,N,4H]
        Z = torch.cat([P, Fv, Nw, Ev], dim=-1)   # [B,N,4H]

        # 4) 公司層級預測 → [B,1,N,1]
        pred_company = self.company_head(Z).unsqueeze(1)  # [B,1,N,1]

        # 5) 整體年度：公司平均特徵 [B,4H] → [B,1,1]
        G = Z.mean(dim=1)                      # [B,4H]
        pred_overall = self.head(G).unsqueeze(1)  # [B,1,1]

        out = {"pred": pred_overall, "pred_company": pred_company}

        # 6) Loss（公司層級 MSE + IC），支援 NaN 遮罩
        if label_company is not None:
            p = pred_company                  # [B,1,N,1]
            t = label_company                 # [B,1,N,1]
            valid = ~torch.isnan(t)
            diff2 = (torch.nan_to_num(p - t, nan=0.0) ** 2) * valid.float()
            # 有效元素數做平均
            denom = valid.float().sum().clamp_min(1.0)
            mse_company = diff2.sum() / denom

            # IC across companies（把 [B,1,N,1] → [B,N]）
            pN = torch.nan_to_num(p.squeeze(1).squeeze(-1), nan=0.0)  # [B,N]
            tN = torch.nan_to_num(t.squeeze(1).squeeze(-1), nan=0.0)  # [B,N]
            if self.ic_type.lower().startswith("pear"):
                ic_per_b = _pearson_corr(pN, tN, dim=1)               # [B]
            else:
                rp = _rank_transform(pN, dim=1)
                rt = _rank_transform(tN, dim=1)
                ic_per_b = _pearson_corr(rp, rt, dim=1)
            ic_loss = ic_per_b.mean()

            out["losses"] = {"mse": mse_company, "ic_company": ic_loss,
                             "total": mse_company }
        return out



