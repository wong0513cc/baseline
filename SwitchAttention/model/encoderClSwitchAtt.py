from typing import Dict, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from model.switchAttention import SwitchMultiModalBlock, PreNorm, MLP, SwitchEncoder
from loss import  _pearson_corr, info_nce_two_modal, multimodal_info_loss_all_pairs
from utils import SafeLayerNorm, PositionalEncoding, head

import numpy as np
import matplotlib.pyplot as plt

# Encoders
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=False):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, 
                            bidirectional=bidirectional, batch_first=True)

    def forward(self, x):
        B, K, N, F = x.shape
        x =x.permute(0,2,1,3).reshape(B*N, K, F)
        self.lstm.flatten_parameters()
        output, _ = self.lstm(x)
        output = output.reshape(B, N, K, -1).permute(0, 2, 1, 3)
        return output
    
# class TransformerEncoder

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
# Cross Atteniotn
class CrossAttention(nn.Module):
    def __init__(self, input_size):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, presentation, qa_session):
        q = self.query(qa_session)      # [B,Lq,H]
        k = self.key(presentation)      # [B,Lp,H]
        v = self.value(presentation)    # [B,Lp,H]
        
        attn_weights = torch.matmul(q, k.transpose(1, 2))  # [B,Lq,Lp]
        attn_weights = self.softmax(attn_weights)
        output = torch.matmul(attn_weights, v)             # [B,Lq,H]
        return output
    
class CrossModalFusion(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.ca_price = CrossAttention(hidden)
        self.ca_fin   = CrossAttention(hidden)
        self.ca_news  = CrossAttention(hidden)
        self.ca_event = CrossAttention(hidden)

    def forward(self, Hp, Hf, Hn, He):
        B, K, N, H = Hp.shape

        # [B,K,N,4,H]
        M = torch.stack([Hp, Hf, Hn, He], dim=3)   #  index
        # 把 (B,K,N) 合併成 batch，modal 當 seq_len: [B*K*N, 4, H]
        M_flat = M.reshape(B * K * N, 4, H)

        def fuse_one(ca_module, idx_self):
            """
            對某一個模態 m:
              - self 嵌入: M_flat[:, idx_self, :]        -> query (Lq=1)
              - 其他模態: M_flat[:, idx_others, :]       -> presentation (Lp=3)
            """
            idx_others = [i for i in range(4) if i != idx_self]

            qa  = M_flat[:, idx_self:idx_self+1, :]   # [B', 1, H]
            prs = M_flat[:, idx_others, :]            # [B', 3, H]

            out = ca_module(presentation=prs, qa_session=qa)  # [B', 1, H]
            out = out.squeeze(1)                              # [B', H]
            return out.reshape(B, K, N, H)                    # [B,K,N,H]

        Hp_f = fuse_one(self.ca_price, 0)   # price ← {fin, news, event}
        Hf_f = fuse_one(self.ca_fin,   1)   # fin   ← {price, news, event}
        Hn_f = fuse_one(self.ca_news,  2)   # news  ← {price, fin, event}
        He_f = fuse_one(self.ca_event, 3)   # event ← {price, fin, news}

        return Hp_f, Hf_f, Hn_f, He_f

    
# Additive Attention
class TemporalAttentionAggregator(nn.Module): # performance更好
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
        Z_year = torch.sum(alpha_time * Z_time, dim=1) # sum over dim=1 (K) -> (B,N,H)
    
        # alpha_time [B,K,N,1] -> [B,N,K]
        alpha_to_viz = alpha_time.squeeze(-1).permute(0, 2, 1)
        return Z_year, alpha_to_viz
    
# MHA    
class TemporalSelfAttention(nn.Module):
    def __init__(self, hidden: int, num_heads: int = 1, dropout: float = 0.1, causal: bool = True):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,  
        )
        self.causal = causal
        self.hidden = hidden

    def _get_causal_mask(self, K: int, device):
        mask = torch.triu(torch.ones(K, K, device=device, dtype=torch.bool), diagonal=1)
        attn_mask = mask.masked_fill(mask, float("-inf")).masked_fill(~mask, 0.0)
        return attn_mask

    def forward(self, Z_time: torch.Tensor):
        B, K, N, H = Z_time.shape
        # [B,K,N,H] -> [B,N,K,H] -> [B*N, K, H]
        x = Z_time.permute(0, 2, 1, 3).reshape(B * N, K, H)

        if self.causal:
            attn_mask = self._get_causal_mask(K, x.device)   # [K,K]
        else:
            attn_mask = None

        # out: [B*N, K, H]
        out, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False) #[B*N, K, H]
        out = out.reshape(B, N, K, H).permute(0, 2, 1, 3) #[b,k,n,h]
        return out, None



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
        # self.enc_price = LSTM(input_size=d_price, hidden_size=hidden, num_layers=1)
        # self.enc_fin = LSTM(input_size=d_finance, hidden_size=hidden, num_layers=1)
        self.enc_price = MLPTimeEncoder(d_in=d_price, hidden=hidden, depth=2, dropout=dropout, K=12)
        self.enc_fin   = MLPTimeEncoder(d_in=d_finance, hidden=hidden, depth=2, dropout=dropout, K=12)
        self.enc_news  = TransformerTimeEncoder(d_in=d_news, d_model=hidden, nhead=nhead_time, num_layers=news_layers, dropout=dropout)
        self.enc_event = MLPTimeEncoder(d_in=d_event, hidden=hidden, depth=2, dropout=dropout, K=12)

        # Attention
        # additive attention
        self.price_additive_attn = TemporalAttentionAggregator(hidden, attn_hidden=128, dropout=dropout)
        self.fin_additive_attn= TemporalAttentionAggregator(hidden, attn_hidden=128, dropout=dropout)
        self.news_additive_attn = TemporalAttentionAggregator(hidden, attn_hidden=128, dropout=dropout)
        self.event_additive_attn = TemporalAttentionAggregator(hidden, attn_hidden=128, dropout=dropout)

        # cross attention
        self.cross_modal = CrossModalFusion(hidden=hidden)

        # multihead attention
        self.self_attn_price = TemporalSelfAttention(hidden=hidden, num_heads=1, dropout=dropout, causal=True)
        self.self_attn_fin = TemporalSelfAttention(hidden=hidden, num_heads=1, dropout=dropout, causal=True)
        self.self_attn_news = TemporalSelfAttention(hidden=hidden, num_heads=1, dropout=dropout, causal=True)
        self.self_attn_event = TemporalSelfAttention(hidden=hidden, num_heads=1, dropout=dropout, causal=True)

        d_fused = hidden*4
        self.predictor = nn.Linear(d_fused, 1)
        # self.predictor = nn.Sequential(nn.LayerNorm(d_fused), nn.Linear(d_fused, d_fused), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_fused, 1),)
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

        # fm = batch.get("finance_mask", None)
        # if fm is not None and fm.dim()==3: fm = fm.unsqueeze(0) # 應對 [K,N,1] -> [B,K,N,1]
        
        # mha
        Hp_sa, _ = self.self_attn_price(Hp)
        Hf_sa, _ = self.self_attn_fin(Hf)
        Hn_sa, _ = self.self_attn_news(Hn)
        He_sa, _ = self.self_attn_event(He)

        # cross modal fusion
        Hp_cm, Hf_cm, Hn_cm, He_cm = self.cross_modal(Hp_sa, Hf_sa, Hn_sa, He_sa)

        # additive attn
        Zp, alpha_p = self.price_additive_attn(Hp_cm)
        Zf, alpha_f = self.fin_additive_attn(Hf_cm)
        Zn, alpha_n = self.news_additive_attn(Hn_cm)
        Ze, alpha_e = self.event_additive_attn(He_cm)
        
        # prediction layer

        # [B,N,H] * 4 -> [B,N, 4*H]
        Z = torch.cat([Zp, Zf, Zn, Ze], dim=-1)
        # Z = torch.cat([Hp_sa[:, -1],Hf_sa[:, -1], Hn_sa[:, -1], He_sa[:, -1]], dim=-1)
        # # Z = torch.cat([Hp_sa.mean(dim=1), Hf_sa.mean(dim=1), Hn_sa.mean(dim=1), He_sa.mean(dim=1)],dim=-1)
        pred_company = self.predictor(Z).unsqueeze(1)  # [B,1,N,1]

        out = {
            "pred_company": pred_company,
            "alpha_time_price": alpha_p,
            "alpha_time_fin": alpha_f,
            "alpha_time_news": alpha_n,
            "alpha_time_event": alpha_e,
            "Z_price": Zp,
            "Z_fin": Zf,
            "Z_news": Zn,
            "Z_event": Ze,
        }


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

            cl_loss = p.new_tensor(0.0)

            # if self.cl_weight > 0.0:
                # cl_loss = multimodal_info_loss_all_pairs(
                #     Zp, Zf, Zn, Ze,
                #     tau=self.cl_tau,
                #     mask_price=None,
                #     mask_fin=(fm.squeeze(0) if (fm is not None and fm.size(0) == 1) else fm),
                #     mask_news=None,
                #     mask_event=None,
                # )

            # total loss：MSE + IC + CL
            total = (
                mse_company
                + self.ic_weight * ((1.0 - ic_company) / 2.0)
                + self.cl_weight * cl_loss
            )

            out["losses"] = {
                "mse": mse_company,
                "ic_company": ic_company,
                "cl": cl_loss,
                "total": total
            }

        return out
    
   
    



