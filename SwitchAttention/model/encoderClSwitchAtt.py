from typing import Dict, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.switchAttention import SwitchMultiModalBlock, PreNorm, MLP, SwitchEncoder
from loss import Projector, multimodal_icl_monthly, _pearson_corr
from utils import SafeLayerNorm, SinusoidalPositionalEncoding, BoundedHead



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
                 icl_weight: float = 0.05,  
                 icl_tau: float = 0.07     
                 ):
        super().__init__()
        self.ic_weight = ic_weight
        self.ic_type = ic_type
        self.icl_weight =icl_weight     
        self.icl_tau = icl_tau           

    
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
        self.predictor = nn.Linear(d_fused, 1)

        # self.proj_price = Projector(in_dim=hidden, hid=128, out_dim=128)
        # self.proj_news  = Projector(in_dim=hidden, hid=128, out_dim=128)
        # self.proj_fin  = Projector(in_dim=hidden, hid=128, out_dim=128)
        # self.proj_event  = Projector(in_dim=hidden, hid=128, out_dim=128)

        # self.deproj_price = nn.Linear(128, hidden)
        # self.deproj_news  = nn.Linear(128, hidden)
        # self.deproj_fin  = nn.Linear(128, hidden)
        # self.deproj_event  = nn.Linear(128, hidden)

        self.proj_price = nn.Identity()
        self.proj_fin   = nn.Identity()
        self.proj_news  = nn.Identity()
        self.proj_event = nn.Identity()

        self.deproj_price = nn.Identity()
        self.deproj_fin   = nn.Identity()
        self.deproj_news  = nn.Identity()
        self.deproj_event = nn.Identity()

        # 每個模態一個 LayerNorm
        self.ln_price = nn.LayerNorm(hidden)
        self.ln_fin   = nn.LayerNorm(hidden)
        self.ln_news  = nn.LayerNorm(hidden)
        self.ln_event = nn.LayerNorm(hidden)

        self.gate_price = nn.Parameter(torch.tensor(0.0))
        self.gate_fin   = nn.Parameter(torch.tensor(0.0))
        self.gate_news  = nn.Parameter(torch.tensor(0.0))
        self.gate_event = nn.Parameter(torch.tensor(0.0))
        
        self.pred_price = Projector(in_dim=hidden, hid=hidden, out_dim=hidden)
        self.pred_fin   = Projector(in_dim=hidden, hid=hidden, out_dim=hidden)
        self.pred_news  = Projector(in_dim=hidden, hid=hidden, out_dim=hidden)
        self.pred_event = Projector(in_dim=hidden, hid=hidden, out_dim=hidden)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        price, finance, news, event = batch["price"], batch["finance"], batch["news"], batch["event"]
        label_company = batch.get("label_company", None)  # [B,1,N,1]
        valid_mask_dict = batch.get("valid_mask_dict", None)  # 可選：{'price':[B,K,N] bool, ...}

        # 1) encoders -> [B,K,N,H]
        Hp = self.enc_price(price)
        Hf = self.enc_fin(finance)
        Hn = self.enc_news(news)
        He = self.enc_event(event)

        B, K, N, H = Hp.shape
        device = Hp.device

        # # 整批投影：H → z
        # zp = self.proj_price(Hp.reshape(-1, H)).reshape(B, K, N, -1)   # [B,K,N,d]
        # zf = self.proj_fin(  Hf.reshape(-1, H)).reshape(B, K, N, -1)
        # zn = self.proj_news( Hn.reshape(-1, H)).reshape(B, K, N, -1)
        # ze = self.proj_event(He.reshape(-1, H)).reshape(B, K, N, -1)
        # d  = zp.size(-1)

        zp, zf, zn, ze = Hp, Hf, Hn, He
        d = H

        # (B) Positive-only CL（逐月×六配對；含遮罩）
        Mask_all = valid_mask_dict if valid_mask_dict is not None else {}
        for k in ("price","finance","news","event"):
            if (k not in Mask_all) or (Mask_all[k] is None):
                Mask_all[k] = torch.ones(B, K, N, dtype=torch.bool, device=device)

        # def pos_only_pair_loss(zA, zB):
        #     # zA,zB: [M', d]（已 L2；正樣本對齊）
        #     if zA.numel() == 0 or zA.size(0) == 0:
        #         return zA.new_tensor(0.0)
        #     cos = F.cosine_similarity(zA, zB, dim=-1)  # [M']
        #     return (1.0 - cos).mean()

        def simsiam_pair_loss(zA: torch.Tensor, zB: torch.Tensor,
                        predA: nn.Module, predB: nn.Module) -> torch.Tensor:
            """
            zA, zB: [M', d]（已經用 mask 篩好）
            predA, predB: 各自模態的 predictor
            """
            if zA.numel() == 0 or zB.numel() == 0:
                return zA.new_tensor(0.0)

            # L2 normalize
            zA = F.normalize(zA, dim=-1)
            zB = F.normalize(zB, dim=-1)

            # predictor 分支
            pA = F.normalize(predA(zA), dim=-1)
            pB = F.normalize(predB(zB), dim=-1)

            # stop-grad 目標
            with torch.no_grad():
                tA = zA.detach()
                tB = zB.detach()

            # SimSiam 對稱 cosine 損失
            loss_ab = 1.0 - F.cosine_similarity(pA, tB, dim=-1)
            loss_ba = 1.0 - F.cosine_similarity(pB, tA, dim=-1)
            return 0.5 * (loss_ab.mean() + loss_ba.mean())

        loss_con_t = []
        for t in range(K):
            Zp_t = zp[:, t].reshape(B*N, d)
            Zf_t = zf[:, t].reshape(B*N, d)
            Zn_t = zn[:, t].reshape(B*N, d)
            Ze_t = ze[:, t].reshape(B*N, d)

            mp = Mask_all['price'][:,  t].reshape(B*N)
            mf = Mask_all['finance'][:,t].reshape(B*N)
            mn = Mask_all['news'][:,  t].reshape(B*N)
            me = Mask_all['event'][:, t].reshape(B*N)

            pairs = [
                # (zA_all, zB_all, mask, predA, predB)
                (Zp_t, Zf_t, mp & mf, self.pred_price, self.pred_fin),   # p↔f
                (Zp_t, Zn_t, mp & mn, self.pred_price, self.pred_news),  # p↔n
                (Zp_t, Ze_t, mp & me, self.pred_price, self.pred_event), # p↔e
                (Zf_t, Zn_t, mf & mn, self.pred_fin,   self.pred_news),  # f↔n
                (Zf_t, Ze_t, mf & me, self.pred_fin,   self.pred_event), # f↔e
                (Zn_t, Ze_t, mn & me, self.pred_news,  self.pred_event), # n↔e
            ]

            lp = []
            for zA_all, zB_all, m, predA, predB in pairs:
                idx = m.nonzero(as_tuple=False).squeeze(1)
                if idx.numel() == 0:
                    continue
                zA = zA_all.index_select(0, idx)  # [M', d] 先用 mask 篩好
                zB = zB_all.index_select(0, idx)
                lp.append(simsiam_pair_loss(zA, zB, predA, predB))

            if lp:
                loss_con_t.append(torch.stack(lp).mean())

        loss_con = torch.stack(loss_con_t).mean() if loss_con_t else Hp.new_tensor(0.0) 



        # 用 z 接回去：deproj + 門控殘差 + LN → 丟進 switch
        ap = torch.sigmoid(self.gate_price)
        af = torch.sigmoid(self.gate_fin)
        an = torch.sigmoid(self.gate_news)
        ae = torch.sigmoid(self.gate_event)

        # # deproj 接收 [..., d]，所以先展平再 reshape 回來
        # Hp_fused = self.ln_price(Hp + ap * self.deproj_price(zp.reshape(-1, d)).reshape(B, K, N, H))
        # Hf_fused = self.ln_fin(  Hf + af * self.deproj_fin(  zf.reshape(-1, d)).reshape(B, K, N, H))
        # Hn_fused = self.ln_news(Hn + an * self.deproj_news( zn.reshape(-1, d)).reshape(B, K, N, H))
        # He_fused = self.ln_event(He + ae * self.deproj_event(ze.reshape(-1, d)).reshape(B, K, N, H))

        Hp_fused = self.ln_price(Hp +ap * zp)
        Hf_fused = self.ln_fin(  Hf +af * zf)
        Hn_fused = self.ln_news(Hn +an * zn)
        He_fused = self.ln_event(He +ae * ze)
        updated_list, fused = self.switch([Hp_fused, Hf_fused, Hn_fused, He_fused])  # switch 需能接受特徵維 d

        # # time pooling -> [B,N,H]
        # P  = updated_list[0].mean(dim=1)
        # Fin  = updated_list[1].mean(dim=1)
        # Nw = updated_list[2].mean(dim=1)
        # E  = updated_list[3].mean(dim=1)

        # # fusion + predict
        # Z = torch.cat([P, Fin, Nw, E], dim=-1) # P,F,E,Nw
        # pred_company = self.company_head(Z).unsqueeze(1)  # [B,1,N,1]

        # 拿最後一個timestep做預測 -> [B,N,H]
        P  = updated_list[0][:, -1, :, :]
        Fin  = updated_list[1][:, -1, :, :]
        Nw = updated_list[2][:, -1, :, :]
        E  = updated_list[3][:, -1, :, :]

        # fusion + predict
        Z = torch.cat([P, Fin, Nw, E], dim=-1) # P,F,E,Nw
        # pred_company = self.company_head(Z).unsqueeze(1)  # [B,1,N,1]
        pred_company = self.predictor(Z).unsqueeze(1)  # [B,1,N,1]


        out = {"pred_company": pred_company}

        if label_company is not None:
            p = pred_company                  # [B,1,N,1]
            t = label_company                 # [B,1,N,1]
            valid = ~torch.isnan(t)
            diff2 = (torch.nan_to_num(p - t, nan=0.0) ** 2) * valid.float()
            denom = valid.float().sum().clamp_min(1.0)
            mse_company = diff2.sum() / denom

            pN = torch.nan_to_num(p.squeeze(1).squeeze(-1), nan=0.0)  # [B,N]
            tN = torch.nan_to_num(t.squeeze(1).squeeze(-1), nan=0.0)  # [B,N]
            ic_per_b = _pearson_corr(pN, tN, dim=1)               # [B]
            ic_company = ic_per_b.mean()

            total = mse_company + self.icl_weight * loss_con + self.ic_weight * ((1.0 - ic_company)/2)
            # total = mse_company

            out["losses"] = {
                "mse": mse_company,
                "ic_company": ic_company,  
                "icl": loss_con,          
                "total": total
            }

        return out





