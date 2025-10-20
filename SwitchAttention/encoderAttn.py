# model.py (with Information Coefficient loss)
from typing import Dict, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        

# --------------------------
# Encoders
# --------------------------
class LSTMAttnEncoder(nn.Module):
    def __init__(self, d_in: int, hidden: int, num_layers: int = 1, bidirectional: bool = False, dropout: float = 0.1, nhead: int = 4, use_posenc: bool = True):
        super().__init__()
        self.proj_in = nn.Linear(d_in, hidden)
        h = hidden // 2 if bidirectional else hidden
        self.lstm = nn.LSTM(input_size=hidden, hidden_size=h, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0)
        self.after_lstm = nn.Linear(h * (2 if bidirectional else 1), hidden) if (h * (2 if bidirectional else 1)) != hidden else nn.Identity()
        self.posenc = SinusoidalPositionalEncoding(hidden) if use_posenc else nn.Identity()
        self.attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden)
        )
        self.norm1 = SafeLayerNorm(hidden)
        self.norm2 = SafeLayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, K, N, _ = x.shape
        x = self.proj_in(x)                                  # [B,K,N,H]
        seq = x.permute(0, 2, 1, 3).reshape(B * N, K, -1)    # [B*N,K,H]
        h, _ = self.lstm(seq)                                 # [B*N,K,*]
        h = self.after_lstm(h)                                # [B*N,K,H]
        h = self.posenc(h.reshape(B, N, K, -1).permute(0, 2, 1, 3).reshape(B * N, K, -1))
        attn_out, _ = self.attn(h, h, h)
        h = self.norm1(h + self.drop(attn_out))
        ff = self.ffn(h)
        h = self.norm2(h + self.drop(ff))
        return h.reshape(B, N, K, -1).permute(0, 2, 1, 3)     # [B,K,N,H]

class TransformerTimeEncoder(nn.Module):
    def __init__(self, d_in: int, d_model: int, nhead: int = 4, num_layers: int = 2, dim_ff: int = 4, dropout: float = 0.1, use_posenc: bool = True):
        super().__init__()
        self.proj_in = nn.Linear(d_in, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=d_model * dim_ff, dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.posenc = SinusoidalPositionalEncoding(d_model) if use_posenc else nn.Identity()
        self.norm = SafeLayerNorm(d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, K, N, _ = x.shape
        x = self.proj_in(x)                                   # [B,K,N,H]
        seq = x.permute(0, 2, 1, 3).reshape(B * N, K, -1)     # [B*N,K,H]
        seq = self.posenc(seq)
        h = self.encoder(seq)                                  # [B*N,K,H]
        h = self.norm(h)
        return h.reshape(B, N, K, -1).permute(0, 2, 1, 3)      # [B,K,N,H]

# --------------------------
# Fusion & Pooling
# --------------------------
class CrossModalFusion(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = SafeLayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.drop = nn.Dropout(dropout)
    def forward(self, Hp: torch.Tensor, Hf: torch.Tensor, Hn: torch.Tensor, He: torch.Tensor) -> torch.Tensor:
        B, K, N, H = Hp.shape
        Z = torch.stack([Hp, Hf, Hn, He], dim=3)  # [B,K,N,4,H]
        Z = Z.reshape(B * K * N, 4, H)
        attn_out, _ = self.attn(Z, Z, Z)
        Z = self.norm(Z + self.drop(attn_out))
        Z = self.norm(Z + self.drop(self.ffn(Z)))
        fused = Z.mean(dim=1)                     # [B*K*N,H]
        return fused.reshape(B, K, N, H)

class CompanyAttentionPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model))
        self.proj = nn.Linear(d_model, d_model, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, K, N, H = x.shape
        q = self.query[None, None, None, :]                 # [1,1,1,H]
        k = self.proj(x)                                    # [B,K,N,H]
        scores = (q * k).sum(dim=-1) / math.sqrt(H)         # [B,K,N]
        attn = torch.softmax(scores, dim=2)                 # [B,K,N]
        pooled = torch.einsum("bkn,bknh->bkh", attn, x)     # [B,K,H]
        return pooled

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

# --------------------------
# Main model
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
                 nhead_cross: int = 4,
                 ic_weight: float = 0.0,           # ← IC loss 權重
                 ic_type: str = "pearson"          # "pearson" 或 "spearman"
                 ):
        super().__init__()
        self.ic_weight = ic_weight
        self.ic_type = ic_type

        # unimodal encoders
        self.enc_price = LSTMAttnEncoder(d_in=d_price, hidden=hidden,
                                         num_layers=lstm_layers, bidirectional=lstm_bidirectional,
                                         dropout=dropout, nhead=nhead_time, use_posenc=True)
        self.enc_fin   = LSTMAttnEncoder(d_in=d_finance, hidden=hidden,
                                         num_layers=lstm_layers, bidirectional=lstm_bidirectional,
                                         dropout=dropout, nhead=nhead_time, use_posenc=True)
        self.enc_news  = TransformerTimeEncoder(d_in=d_news, d_model=hidden,
                                                nhead=nhead_time, num_layers=news_layers, dim_ff=4, dropout=dropout, use_posenc=True)
        self.enc_event = TransformerTimeEncoder(d_in=d_event, d_model=hidden,
                                                nhead=nhead_time, num_layers=event_layers, dim_ff=4, dropout=dropout, use_posenc=True)

        # cross-modal fusion at (t,n)
        self.fusion = CrossModalFusion(d_model=hidden, nhead=nhead_cross, dropout=dropout)

        # company pooling (over N) per time step
        self.pool = CompanyAttentionPool(d_model=hidden)

        # prediction head (per time step)
        self.company_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            SafeLayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            SafeLayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
        self.head_min = 0.0     # 如果你的標籤是 0~100，改成 0.0
        self.head_max = 1.0     # 如果你的標籤是 0~100，改成 100.0

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


    # def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    #     price, finance, news, event = batch["price"], batch["finance"], batch["news"], batch["event"]
    #     label_overall = batch.get("label_overall", None)  # [B,1,1]
    #     label_company = batch.get("label_company", None)  # [B,1,N,1]

    #     # 1) unimodal encoders -> [B,K,N,H]
    #     Hp = self.enc_price(price); Hf = self.enc_fin(finance); Hn = self.enc_news(news); He = self.enc_event(event)

    #     # 2) cross-modal fusion -> [B,K,N,H]
    #     H_fused = self.fusion(Hp, Hf, Hn, He)

    #     # # 3) 先「時間池化」得到逐公司年度表示 [B,1,N,H]
    #     # H_company_year = H_fused.mean(dim=1, keepdim=True)  # 可換成注意力或 last

    #     # # 4) 逐公司年度預測 [B,1,N,1]
    #     # pred_company = self.company_head(H_company_year)

    #     # # 5) 公司池化 → 整體年度表示 [B,1,H]，再 head → [B,1,1]
    #     # #    （沿 N 用 attention/mean，這裡復用 CompanyAttentionPool：餵入 [B,1,N,H]）
    #     # G_year = self.pool(H_company_year)          # [B,1,H]
    #     # pred_overall = self.head(G_year)            # [B,1,1]

    #     # 3) 時間池化到逐公司年度表示 [B,N,H]
    #     Z_company = H_fused.mean(dim=1)  # K 維做平均

    #     # 4) 公司層預測（頭吃 hidden）：[B,N,1] → [B,1,N,1]
    #     pred_company = self.company_head(Z_company).unsqueeze(1)

    #     # 5) 公司池化做年度整體（沿 N）：先把 H_fused 做時間平均→[B,1,N,H]→pool→[B,1,H]
    #     H_year_step = H_fused.mean(dim=1, keepdim=True)  # [B,1,N,H]
    #     G_year = self.pool(H_year_step)                  # [B,1,H]
    #     pred_overall = self.head(G_year).unsqueeze(1)    # [B,1,1]

    #     out = {"pred": pred_overall, "pred_company": pred_company,
    #         "features": {"fused": H_fused, "company_year": Z_company, "pooled_year": G_year}}




    #     # 6) 損失：公司層級 MSE + 公司 IC（沿 N 算 corr）
    #     # 期望：label_company ∈ [B,1,N,1]；pred_company ∈ [B,1,N,1]
    #     if label_company is not None:
    #         losses = {}

    #         # --- Company-level MSE ---
    #         p = pred_company                                   # [B,1,N,1]
    #         t = torch.nan_to_num(label_company, nan=0.0)       # 同形狀，處理 NaN
    #         # pred_company, label_company: [B,1,N,1]
    #         se = (pred_company - label_company).pow(2)          # [B,1,N,1]
    #         sse_per_sample = se.sum(dim=(1,2,3))                # [B]：每個樣本把 N 全部加總
    #         mse_company = sse_per_sample.mean()                        # 會自動在所有維度平均
    #         losses["mse"] = mse_company

    #         # --- Company-level IC (Pearson 或 Spearman) ---
    #         pN = p.squeeze(1).squeeze(-1)   # [B,N]
    #         tN = t.squeeze(1).squeeze(-1)   # [B,N]
    #         if self.ic_type.lower().startswith("pear"):
    #             ic_per_b = _pearson_corr(pN, tN, dim=1)        # [B]
    #         else:
    #             rp = _rank_transform(pN, dim=1)                # Spearman: 先轉名次
    #             rt = _rank_transform(tN, dim=1)
    #             ic_per_b = _pearson_corr(rp, rt, dim=1)        # [B]
    #         ic_loss = -ic_per_b.mean()
    #         losses["ic_company"] = ic_loss

    #         # --- Total ---
    #         losses["total"] = losses["mse"] + self.ic_weight * losses["ic_company"]
    #         out["losses"] = losses

    #     return out


    def forward(self, batch):
        price, finance, news, event = batch["price"], batch["finance"], batch["news"], batch["event"]
        label_company = batch.get("label_company", None)

        # 1) unimodal encoders -> [B,K,N,H]
        Hp = self.enc_price(price)
        Hf = self.enc_fin(finance)
        Hn = self.enc_news(news)
        He = self.enc_event(event)

        # 2) cross-modal fusion -> [B,K,N,H]
        H_fused = self.fusion(Hp, Hf, Hn, He)

        # 3) 時間聚合到逐公司年度向量 [B,N,H]
        Z_company = H_fused.mean(dim=1)  # 沿 K 平均

        # 4) 公司層預測 [B,N,1] → 壓界 → [B,1,N,1]
        raw_company = self.company_head(Z_company)            # [B,N,1]
        pred_company = torch.sigmoid(raw_company) * (self.head_max - self.head_min) + self.head_min
        pred_company = pred_company.unsqueeze(1)              # [B,1,N,1]

        # 5) 整體年度：先時序平均成 [B,1,N,H]，池化公司得到 [B,1,H] → head → 壓界 → [B,1,1]
        H_year_step = H_fused.mean(dim=1, keepdim=True)       # [B,1,N,H]
        G_year = self.pool(H_year_step)                       # [B,1,H]
        raw_overall = self.head(G_year)                       # [B,1,1]
        pred_overall = torch.sigmoid(raw_overall) * (self.head_max - self.head_min) + self.head_min

        out = {"pred": pred_overall, "pred_company": pred_company,
            "features": {"fused": H_fused, "company": Z_company, "pooled_year": G_year}}

        # 6) Loss（遮罩版 MSE + IC）——若你已經有這段就不用改
        if label_company is not None:
            p = pred_company                  # [B,1,N,1]
            t = label_company                 # [B,1,N,1]
            valid = ~torch.isnan(t)
            diff2 = (torch.nan_to_num(p - t, nan=0.0)**2) * valid.float()
            denom = valid.float().sum().clamp_min(1.0)
            mse_company = diff2.sum() / denom

            pN = torch.nan_to_num(p.squeeze(1).squeeze(-1), nan=0.0)  # [B,N]
            tN = torch.nan_to_num(t.squeeze(1).squeeze(-1), nan=0.0)  # [B,N]
            if self.ic_type.lower().startswith("pear"):
                ic_per_b = _pearson_corr(pN, tN, dim=1)
            else:
                rp = _rank_transform(pN, dim=1)
                rt = _rank_transform(tN, dim=1)
                ic_per_b = _pearson_corr(rp, rt, dim=1)
            ic_loss = ic_per_b.mean()

            out["losses"] = {"mse": mse_company,
                            "ic_company": ic_loss,
                            "total": mse_company}
        return out


