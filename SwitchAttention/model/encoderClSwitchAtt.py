from typing import Dict, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from model.switchAttention import SwitchMultiModalBlock, PreNorm, MLP, SwitchEncoder
from loss import Projector, multimodal_icl_monthly, _pearson_corr
from utils import SafeLayerNorm, PositionalEncoding, head

import numpy as np
import matplotlib.pyplot as plt


# Encoders
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
        self.posenc = PositionalEncoding(hidden) if use_posenc else nn.Identity()
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
        self.posenc = PositionalEncoding(d_model) if use_posenc else nn.Identity()
        self.norm = SafeLayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, K, N, _ = x.shape
        x = self.proj_in(x)                                   # [B,K,N,H]
        seq = x.permute(0, 2, 1, 3).reshape(B * N, K, -1)     # [B*N,K,H]
        seq = self.posenc(seq)
        h = self.encoder(seq)                                 # [B*N,K,H]
        h = self.norm(h)
        return h.reshape(B, N, K, -1).permute(0, 2, 1, 3)     # [B,K,N,H]
    

class MLPTimeEncoder(nn.Module):
    def __init__(self, d_in, hidden, depth = 2, dropout: float = 0.1, K: int = 12):
        super().__init__()
        layers = [nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)]
        self.ff = nn.Sequential(*layers)
        self.ln = nn.LayerNorm(hidden)
        self.time_emb = nn.Embedding(K, hidden)

    def forward(self, x):  # x: [B,K,N,D_in]    
        B,K,N,D = x.shape
        y = self.ff(x)                       # [B,K,N,H]
        t = torch.arange(K, device=x.device)  # [K]
        te = self.time_emb(t).view(1,K,1,-1)  # [1,K,1,H]
        y = y + te
        y = self.ln(y)
        return y       
    
# Attention    
# class TemporalAggregator(nn.Module):
#     def __init__(self, hidden, num_heads=4, dropout=0.1):
#         super().__init__()
#         self.attn = nn.MultiheadAttention(
#             embed_dim=hidden, num_heads=num_heads,
#             dropout=dropout, batch_first=True
#         )

#     def forward(self, Z_time):  # [B,K,N,H]
#         B,K,N,H = Z_time.shape
#         # 每家公司單獨跑一次 attention
#         Z_time = Z_time.permute(0,2,1,3).reshape(B*N, K, H)  # [B*N,K,H]

#         # 用一個 learnable 的 global query 來 pool
#         # 這裡簡化：把第一個 time step 當 query，其實可以改成專門的 q 向量
#         q = Z_time[:, :1, :]         # [B*N,1,H]
#         k = v = Z_time               # [B*N,K,H]

#         z, attn = self.attn(q, k, v) # z:[B*N,1,H], attn:[B*N,1,K]
#         z = z.squeeze(1)             # [B*N,H]
#         attn = attn.squeeze(1)       # [B*N,K]

#         z = z.reshape(B,N,H)
#         attn = attn.reshape(B,N,K)
#         return z, attn
    
# class TemporalAggregator(nn.Module):
#     def __init__(self, hidden, num_heads=4, dropout=0.1):
#         super().__init__()
#         self.attn = nn.MultiheadAttention(
#             embed_dim=hidden,
#             num_heads=num_heads,
#             dropout=dropout,
#             batch_first=True,     # 所以 attn 的 input 是 [B, L, H]
#         )
#         # 專門的「global query」：[1, 1, H]
#         self.query = nn.Parameter(torch.randn(1, 1, hidden))

#     def forward(self, Z_time, mask: torch.Tensor = None):
#         """
#         Z_time: [B,K,N,H]
#         mask:   [B,K,N,1] or [B,K,N]，1=有效、0=padding（選用）
#         回傳:
#           z:     [B,N,H]
#           attn:  [B,N,K]  （每家公司對每個月份的權重）
#         """
#         B, K, N, H = Z_time.shape

#         # [B,K,N,H] -> [B,N,K,H] -> [B*N,K,H]
#         seq = Z_time.permute(0, 2, 1, 3).reshape(B * N, K, H)   # K = time steps

#         # ==== 準備 query ====
#         # self.query: [1,1,H]  -> 複製成每家公司一個 query
#         q = self.query.expand(B * N, 1, H)                      # [B*N,1,H]

#         # MultiheadAttention 的 key_padding_mask 形狀是 [batch, seq_len] = [B*N, K]
#         key_padding_mask = None
#         if mask is not None:
#             # 先變成 [B,K,N]
#             if mask.dim() == 4:          # [B,K,N,1]
#                 mask = mask.squeeze(-1)
#             # mask: 1=有效 → key_padding_mask 要求  True=要遮 → 所以取反
#             # 先整理成 [B,N,K] 再 flatten
#             mask_bnK = mask.permute(0, 2, 1)        # [B,N,K]
#             key_padding_mask = (mask_bnK.reshape(B * N, K) == 0)  # bool [B*N,K]

#         # ==== Attention ====
#         # q:   [B*N,1,H]
#         # k,v: [B*N,K,H]
#         # key_padding_mask: [B*N,K]，True 代表該位置被遮
#         z, attn = self.attn(
#             q, seq, seq,
#             key_padding_mask=key_padding_mask
#         )                           # z: [B*N,1,H], attn: [B*N,1,K]

#         z = z.squeeze(1)            # [B*N,H]
#         attn = attn.squeeze(1)      # [B*N,K]

#         # 還原成 [B,N,H] [B,N,K]
#         z = z.reshape(B, N, H)
#         attn = attn.reshape(B, N, K)
#         return z, attn

class TemporalAttentionAggregator(nn.Module):
    """
    將 Z_time ∈ [B,K,N,H] 聚合為 Z_year ∈ [B,N,H]
    """
    def __init__(self, hidden: int, attn_hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, attn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(attn_hidden, 1)
        )

    def forward(self, Z_time: torch.Tensor, mask: torch.Tensor = None):
        # Z_time: [B,K,N,H]
        scores = self.scorer(Z_time) #(B,K,N,1)
    
        if mask is not None:
            scores = scores.masked_fill(mask <= 0, float('-inf'))
            
        alpha_time = F.softmax(scores, dim=1) # [B,K,N,1]
        
        # (B,K,N,1) * (B,K,N,H) -> (B,K,N,H)
        # sum over dim=1 (K) -> (B,N,H)
        Z_year = torch.sum(alpha_time * Z_time, dim=1)
    
        # alpha_time [B,K,N,1] -> [B,N,K]
        alpha_to_viz = alpha_time.squeeze(-1).permute(0, 2, 1)
        
        return Z_year, alpha_to_viz



# Main model
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
                 ic_weight: float = 0.1,   
                 ic_type: str = "pearson",
                 cl_weight: float = 0.0,   
                 cl_tau: float = 0.07,
                 ):
        super().__init__()
        self.ic_weight = ic_weight
        self.ic_type = ic_type
        self.cl_weight = cl_weight
        self.cl_tau = cl_tau

        # Encoders
        self.enc_price = MLPTimeEncoder(d_in=d_price, hidden=hidden, depth=2, dropout=dropout, K=12)
        self.enc_fin   = MLPTimeEncoder(d_in=d_finance, hidden=hidden, depth=2, dropout=dropout, K=12)
        self.enc_news  = TransformerTimeEncoder(d_in=d_news,  d_model=hidden, nhead=nhead_time, num_layers=news_layers,  dropout=dropout)
        self.enc_event = MLPTimeEncoder(d_in=d_event, hidden=hidden, depth=2, dropout=dropout, K=12)

        self.price_attn = TemporalAttentionAggregator(hidden, attn_hidden=128, dropout=dropout)
        self.fin_attn= TemporalAttentionAggregator(hidden, attn_hidden=128, dropout=dropout)
        self.news_attn = TemporalAttentionAggregator(hidden, attn_hidden=128, dropout=dropout)
        self.event_attn = TemporalAttentionAggregator(hidden, attn_hidden=128, dropout=dropout)

        # self.price_attn = TemporalAggregator(hidden=hidden, dropout=dropout)
        # self.fin_attn = TemporalAggregator(hidden=hidden, dropout=dropout)
        # self.news_attn = TemporalAggregator(hidden=hidden, dropout=dropout)
        # self.event_attn = TemporalAggregator(hidden=hidden, dropout=dropout)

        d_fused = hidden * 4
        self.predictor = nn.Linear(d_fused, 1)
        self.ln = nn.LayerNorm(hidden)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        price, finance, news, event = batch["price"], batch["finance"], batch["news"], batch["event"]
        adj = batch["network"]                           # [B,K,N,N] bool/0-1
        label_company = batch.get("label_company", None) # [B,1,N,1]

        # encoders [B,K,N,H]
        Hp = self.enc_price(price)
        Hf = self.enc_fin(finance)
        Hn = self.enc_news(news)
        He = self.enc_event(event)
        
        fm = batch.get("finance_mask", None)
        if fm is not None and fm.dim()==3: fm = fm.unsqueeze(0) # 應對 [K,N,1] -> [B,K,N,1]

        # # [B,K,N,H] -> [B,N,H]
        # Zp, alpha_p = self.price_attn(Hp, mask=None) 
        # Zf, alpha_f = self.fin_attn(Hf, mask=fm) 
        # Zn, alpha_n = self.news_attn(Hn, mask=None)
        # Ze, alpha_e = self.event_attn(He, mask=None)

        Zp, alpha_p = self.price_attn(Hp) 
        Zf, alpha_f = self.fin_attn(Hf) 
        Zn, alpha_n = self.news_attn(Hn)
        Ze, alpha_e = self.event_attn(He)



        # [B,N,H] * 4 -> [B,N, 4*H]
        Z = torch.cat([Zp, Zf, Zn, Ze], dim=-1)
        pred_company = self.predictor(Z).unsqueeze(1)  # [B,1,N,1]

        out = {"pred_company": pred_company}
        
        out["alpha_time_price"] = alpha_p
        out["alpha_time_fin"] = alpha_f
        out["alpha_time_news"] = alpha_n
        out["alpha_time_event"] = alpha_e

        if label_company is not None:
            p = pred_company                  # [B,1,N,1]
            t = label_company                 # [B,1,N,1]
            valid = ~torch.isnan(t)
            diff2 = (torch.nan_to_num(p - t, nan=0.0) ** 2) * valid.float()
            denom = valid.float().sum().clamp_min(1.0)
            mse_company = diff2.sum() / denom

            # IC company-level
            pN = torch.nan_to_num(p.squeeze(1).squeeze(-1), nan=0.0)  # [B,N]
            tN = torch.nan_to_num(t.squeeze(1).squeeze(-1), nan=0.0)  # [B,N]
            ic_per_b = _pearson_corr(pN, tN, dim=1)                    # [B]
            ic_company = ic_per_b.mean()

            total = (
                mse_company
                + self.ic_weight * ((1.0 - ic_company) / 2.0)
            )

            out["losses"] = {
                "mse": mse_company,
                "ic_company": ic_company,
                "total": total
            }
            
        return out
    
   
    



