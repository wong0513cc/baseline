from typing import Dict, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from model.switchAttention import SwitchMultiModalBlock, PreNorm, MLP, SwitchEncoder
from loss import  _pearson_corr, info_nce_two_modal, multimodal_info_loss_all_pairs, _triplet_loss_batch_all, multimodal_triplet_loss_4modal
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
    
# TransformerEncoder
class TransformerTimeEncoder(nn.Module):
    def __init__(self, d_in: int, d_model: int, nhead: int = 4, num_layers: int = 2, dim_ff: int = 4, dropout: float = 0.1, use_posenc: bool = True):
        super().__init__()
        self.proj_in = nn.Linear(d_in, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * dim_ff, dropout=dropout, batch_first=True, norm_first=True)
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

    def forward(self, q, k, v):
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        
        attn_weights = torch.matmul(q, k.transpose(1, 2))
        attn_weights = self.softmax(attn_weights)
        output = torch.matmul(attn_weights, v)
        
        return output

    
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
    

class SwitchLayer(nn.Module):
    """
    一層 switch-attention：
    - 只更新一個模態 (q_mod)
    - Q 來自 q_mod，K/V 來自其他模態
    - 用共享 CrossAttention（共用 QKV 權重）
    """
    def __init__(self, hidden: int, q_mod: str, k_mod: str, v_mod: str,
                 ca: CrossAttention, dropout: float = 0.1, residual_scale: float = 0.5):
        super().__init__()
        self.ca = ca                     # 共用的一個 CrossAttention
        self.q_mod = q_mod               # 'p', 'f', 'n', 'e'
        self.k_mod = k_mod
        self.v_mod = v_mod
        self.s = residual_scale

        self.ln1 = nn.LayerNorm(hidden)
        self.ff  = nn.Sequential(
            nn.Linear(hidden, 4 * hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden, hidden),
        )
        self.ln2 = nn.LayerNorm(hidden)

    def forward(self, Zp, Zf, Zn, Ze):
        # 取出目前四個模態的表示
        d = {'p': Zp, 'f': Zf, 'n': Zn, 'e': Ze}
        q = d[self.q_mod]   # [B,N,H]
        k = d[self.k_mod]   # [B,N,H]
        v = d[self.v_mod]   # [B,N,H]

        # Cross-attn 更新 q 這個模態
        delta = self.ca(q, k, v)         # [B,N,H]
        q_new = self.ln1(q + self.s * delta)

        # FFN
        ff_out = self.ff(q_new)
        q_new = self.ln2(q_new + ff_out)

        # 寫回對應的那個模態
        if self.q_mod == 'p':
            Zp = q_new
        elif self.q_mod == 'f':
            Zf = q_new
        elif self.q_mod == 'n':
            Zn = q_new
        elif self.q_mod == 'e':
            Ze = q_new

        return Zp, Zf, Zn, Ze

class SwitchAttention(nn.Module):
    """
    更貼近論文的 Switch-Attention：
    - 內含 4 個 SwitchLayer
    - 每層只更新一個模態，Q/K/V 來源依序為：
        1) Q=P, K=F, V=N
        2) Q=F, K=N, V=E
        3) Q=N, K=E, V=P
        4) Q=E, K=P, V=F
    - 四層都共用同一個 CrossAttention，所以 Q/K/V projection 在所有模態 & 層共享
    """
    def __init__(self, hidden: int, dropout: float = 0.1, residual_scale: float = 0.5):
        super().__init__()
        ca = CrossAttention(hidden)   # 一個共享的 QKV

        self.layers = nn.ModuleList([
            SwitchLayer(hidden, 'p', 'f', 'n', ca, dropout, residual_scale),
            SwitchLayer(hidden, 'f', 'n', 'e', ca, dropout, residual_scale),
            SwitchLayer(hidden, 'n', 'e', 'p', ca, dropout, residual_scale),
            SwitchLayer(hidden, 'e', 'p', 'f', ca, dropout, residual_scale),
        ])

    def forward(self, Zp, Zf, Zn, Ze):
        for layer in self.layers:
            Zp, Zf, Zn, Ze = layer(Zp, Zf, Zn, Ze)
        return Zp, Zf, Zn, Ze


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
                 freeze_enc: bool = True
                 ):
        super().__init__()
        self.ic_weight = ic_weight
        self.ic_type = ic_type
        self.cl_weight = cl_weight
        self.cl_tau = cl_tau

        # Encoders
        self.enc_price = MLPTimeEncoder(d_in=d_price, hidden=hidden, depth=2, dropout=dropout, K=12)
        self.enc_fin   = MLPTimeEncoder(d_in=d_finance, hidden=hidden, depth=2, dropout=dropout, K=12)
        self.enc_news  = TransformerTimeEncoder(d_in=d_news, d_model=hidden, nhead=nhead_time, num_layers=news_layers, dropout=dropout)
        self.enc_event = MLPTimeEncoder(d_in=d_event, hidden=hidden, depth=2, dropout=dropout, K=12)

        # multihead attention
        self.self_attn_price = TemporalSelfAttention(hidden=hidden, num_heads=1, dropout=dropout, causal=True)
        self.self_attn_fin = TemporalSelfAttention(hidden=hidden, num_heads=1, dropout=dropout, causal=True)
        self.self_attn_news = TemporalSelfAttention(hidden=hidden, num_heads=1, dropout=dropout, causal=True)
        self.self_attn_event = TemporalSelfAttention(hidden=hidden, num_heads=1, dropout=dropout, causal=True)

        # Attention
        # additive attention
        self.price_additive_attn = TemporalAttentionAggregator(hidden, attn_hidden=128, dropout=dropout)
        self.fin_additive_attn= TemporalAttentionAggregator(hidden, attn_hidden=128, dropout=dropout)
        self.news_additive_attn = TemporalAttentionAggregator(hidden, attn_hidden=128, dropout=dropout)
        self.event_additive_attn = TemporalAttentionAggregator(hidden, attn_hidden=128, dropout=dropout)

        # fusion
        self.switch_attn = SwitchAttention(hidden=hidden, dropout=dropout, residual_scale=0.5)

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
        if fm is not None and fm.dim()==3: fm = fm.unsqueeze(0) # 應對 [K,N,1] -> [B,K,N,1

        # mha
        Hp_sa, _ = self.self_attn_price(Hp)
        Hf_sa, _ = self.self_attn_fin(Hf)
        Hn_sa, _ = self.self_attn_news(Hn)
        He_sa, _ = self.self_attn_event(He)

        # additive attn
        Zp, alpha_p = self.price_additive_attn(Hp_sa)
        Zf, alpha_f = self.fin_additive_attn(Hf_sa, mask = fm)
        Zn, alpha_n = self.news_additive_attn(Hn_sa)
        Ze, alpha_e = self.event_additive_attn(He_sa)

        # cross modal fusion
        Zp, Zf, Zn, Ze = self.switch_attn(Zp, Zf, Zn, Ze)

        # prediction layer
        # [B,N,H] * 4 -> [B,N, 4*H]

        Z = torch.cat([Zp, Zf, Zn, Ze], dim=-1)
        # Z = torch.cat([Hp_aligned[:, -1],Hf_aligned[:, -1], Hn_aligned[:, -1], He_aligned[:, -1]], dim=-1)
        # Z = torch.cat([Hp_aligned.mean(dim=1), Hf_aligned.mean(dim=1), Hn_aligned.mean(dim=1), He_aligned.mean(dim=1)],dim=-1)

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
            if self.cl_weight > 0.0:
                # Zp, Zf, Zn, Ze: [B,N,H]
                embeds = {
                    "price": Zp,
                    "fin":   Zf,
                    "news":  Zn,
                    "event": Ze,
                }

                # 確保 batch 裡有 company_id: [B,N]，例如來自 yearly symbol 的 idx
                company_id = batch["company_id"]       # [B,N] int

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
    
   
    



