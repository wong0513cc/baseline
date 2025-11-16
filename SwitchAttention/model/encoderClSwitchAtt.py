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
        self.posenc = PositionalEncoding(d_in) if use_posenc else nn.Identity()
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
class TemporalAttentionPool(nn.Module):
    def __init__(self, hidden, attn_hidden=128, dropout=0.1):
        super().__init__()
        # 先算attention score
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden), 
            nn.Linear(hidden, attn_hidden), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(attn_hidden, 1)
        )

    def forward(self, X, mask=None):
        B,K,N,Hd = X.shape
        score = self.scorer(X).squeeze(-1)      # [B,K,N]
        if mask is not None:
            m = mask.squeeze(-1).float()        # [B,K,N]
            score = score.masked_fill(m <= 0, float('-inf'))
        alpha = score.softmax(dim=1)
        z = (alpha.unsqueeze(-1) * X).sum(dim=1)  # [B,N,H]
        return z, alpha.permute(0,2,1) 
    

class TemporalSelfAttention(nn.Module):
    def __init__(self, hidden, nhead=4, num_layers=1, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead,
            dim_feedforward=hidden*4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(hidden)

    def forward(self, H, attn_mask=None, key_padding_mask=None):
        # H: [B,K,N,H]
        B,K,N,Hd = H.shape
        x = H.permute(0,2,1,3).reshape(B*N, K, Hd)   # [B*N,K,H]
        y = self.enc(x, mask=attn_mask, src_key_padding_mask=key_padding_mask)  # [B*N,K,H]
        y = self.ln(y)
        return y.reshape(B, N, K, Hd).permute(0,2,1,3)  # [B,K,N,H]
           

class GatedAttention(nn.Module):
    def __init__(self, hidden, attn_hidden=128, dropout=0.1, modality_order=None):
        super().__init__()
        self.hidden = hidden
        self.modality_order = modality_order or ["price", "finance", "news", "event"]
        self.num_modalities = len(self.modality_order)
        self.ga = nn.Sequential(
            nn.LayerNorm(self.num_modalities * hidden),
            nn.Linear(self.num_modalities * hidden, attn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(attn_hidden, self.num_modalities)
        )
    def forward(self, H_dict: dict, mask_dict: dict = None):
        Hs = [H_dict[m] for m in self.modality_order]  # list of [B,K,N,H]
        B, K, N, H = Hs[0].shape
        global_context = torch.cat(Hs, dim=-1)
        scores = self.ga(global_context)

        if mask_dict is not None:
            masks = []
            for m in self.modality_order:
                mm = mask_dict.get(m, None)
                if mm is None:
                    mm = torch.ones((B, K, N, 1), device=global_context.device, dtype=torch.float32)
                masks.append(mm)
            MM = torch.stack(masks, dim=3).squeeze(-1)
            scores = scores.masked_fill(MM <= 0, float('-inf'))
        
        alpha = F.softmax(scores, dim=-1)

        stack_embeddings = torch.stack(Hs, dim=3)
        weights_reshaped = alpha.unsqueeze(-1)
        Z = torch.sum(stack_embeddings * weights_reshaped, dim=3)

        return Z, alpha


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
        # self.enc_price = LSTMTimeEncoder(d_in=d_price, hidden=hidden,
        #                                  num_layers=lstm_layers, bidirectional=lstm_bidirectional,
        #                                  dropout=dropout, use_posenc=False)
        # self.enc_fin   = LSTMTimeEncoder(d_in=d_finance, hidden=hidden,
        #                                  num_layers=lstm_layers, bidirectional=lstm_bidirectional,
        #                                  dropout=dropout, use_posenc=False)
        self.enc_news  = TransformerTimeEncoder(d_in=d_news,  d_model=hidden,
                                                nhead=nhead_time, num_layers=news_layers, dropout=dropout)
        self.enc_event = TransformerTimeEncoder(d_in=d_event, d_model=hidden,
                                                nhead=nhead_time, num_layers=event_layers, dropout=dropout)
        #mlp Encoders
        self.enc_price = MLPTimeEncoder(d_in=d_price, hidden=hidden, depth=2, dropout=dropout, K=12)
        self.enc_fin   = MLPTimeEncoder(d_in=d_finance, hidden=hidden, depth=2, dropout=dropout, K=12)
        # self.enc_news  = MLPTimeEncoder(d_in=d_news,   hidden=hidden, depth=news_layers, dropout=dropout, K=12)
        # self.enc_event = MLPTimeEncoder(d_in=d_event,  hidden=hidden, depth=event_layers, dropout=dropout, K=12)

        d_fused = hidden
        self.predictor = nn.Linear(d_fused, 1)
        self.ln = nn.LayerNorm(hidden)

        self.ga = GatedAttention(hidden=hidden, attn_hidden=128, dropout=0.1, modality_order=["price","finance","news","event"])
        self.temp_agg = TemporalAttentionPool(hidden=hidden, attn_hidden=128, dropout=0.1)
        self.temp_selfattn = TemporalSelfAttention(hidden=hidden, nhead=4, num_layers=1, dropout=0.1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        price, finance, news, event = batch["price"], batch["finance"], batch["news"], batch["event"]
        adj = batch["network"]                           # [B,K,N,N] bool/0-1
        label_company = batch.get("label_company", None) # [B,1,N,1]

        # encoders [B,K,N,H]
        Hp = self.enc_price(price)
        Hf = self.enc_fin(finance)
        Hn = self.enc_news(news)
        He = self.enc_event(event)

        # attention 
        Hp = self.temp_selfattn(Hp)   # [B,K,N,H]
        Hf = self.temp_selfattn(Hf)
        Hn = self.temp_selfattn(Hn)
        He = self.temp_selfattn(He)

        H_dict = {"price":Hp, "finance":Hf, "news":Hn, "event":He}
        mask_dict = {"price":None, "news":None, "event":None}
        fm = batch.get("finance_mask", None)
        if fm is not None and fm.dim()==3: fm = fm.unsqueeze(0)
        mask_dict["finance"] = fm

        Z_time, alpha = self.ga(H_dict, mask_dict=mask_dict)  # [B,K,N,H]
        Z_year, alpha_time   = self.temp_agg(Z_time, mask=None) 
        print(alpha_time)

        # Z = torch.cat([
        #     Hp[:, -1],   # [B,N,H]
        #     Hf[:, -1],
        #     Hn[:, -1],
        #     He[:, -1],
        # ], dim=-1)  # [B,N, 4H]

        # Z = torch.cat([zp, zf, zn, ze], dim=-1)  # [B,N,4H]
  
        pred_company = self.predictor(Z_year).unsqueeze(1)  # [B,1,N,1]
        out = {"pred_company": pred_company}

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
    
   
    



