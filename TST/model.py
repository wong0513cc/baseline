
# esg_tst_baseline.py

import torch
import torch.nn as nn
from typing import Dict

from loss import _pearson_corr
from tst import TSTransformerEncoderClassiregressor 


class ESG_TSTBaseline(nn.Module):
    """
    多模態 concat + 原版 TST (TSTransformerEncoderClassiregressor) 的 ESG baseline

    Input batch keys:
        - price:   [B,12,N,Dp]
        - finance: [B,12,N,Df]
        - news:    [B,12,N,Dn]
        - event:   [B,12,N,De]
        - label_company: [B,1,N,1]  (0~1, 已除以 100)

    Output:
        out = {
            "pred_company": [B,1,N,1],
            "losses": { "mse", "ic_company", "total" }
        }
    """

    def __init__(
        self,
        d_price: int,
        d_finance: int,
        d_news: int,
        d_event: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pos_encoding: str = "learnable",  
        activation: str = "gelu",
        norm: str = "BatchNorm",
        ic_weight: float = 0.1,
    ):
        super().__init__()

        self.ic_weight = ic_weight

        self.K = 12
        self.feat_dim = d_price + d_finance + d_news + d_event

        self.tst = TSTransformerEncoderClassiregressor(
            feat_dim=self.feat_dim,
            max_len=self.K,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            num_classes=1,            # 回傳一個 scalar，regression 用
            dropout=dropout,
            pos_encoding=pos_encoding,
            activation=activation,
            norm=norm,
            freeze=False,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        price   = batch["price"]    # [B,12,N,Dp]
        finance = batch["finance"]  # [B,12,N,Df]
        news    = batch["news"]     # [B,12,N,Dn]
        event   = batch["event"]    # [B,12,N,De]
        label_company = batch.get("label_company", None)  # [B,1,N,1]

        # 1) 多模態 concat -> [B,12,N, D_all]
        X = torch.cat([price, finance, news, event], dim=-1)
        B, K, N, D_all = X.shape
        assert K == self.K, f"expect K={self.K}, got {K}"
        assert D_all == self.feat_dim, f"feat dim mismatch: {D_all} vs {self.feat_dim}"

        # 2) 併公司到 batch 維度: [B,N,K,D] -> [B*N,K,D]
        X_seq = X.permute(0, 2, 1, 3).reshape(B * N, K, D_all)  # [B*N,12,D_all]

        # 3) padding mask：你沒有 variable length，就全 True
        padding_masks = torch.ones(B * N, K, dtype=torch.bool, device=X_seq.device)

        # 4) 丟進 TST：output: [B*N, 1]
        out_flat = self.tst(X_seq, padding_masks)      # [B*N,1]

        # 5) reshape 回 [B,1,N,1]
        pred_company = out_flat.view(B, N, 1).permute(0, 2, 1).unsqueeze(-1)  # [B,1,N,1]

        out = {
            "pred_company": pred_company,
        }

        if label_company is not None:
            p = pred_company
            t = label_company

            valid = ~torch.isnan(t)
            diff2 = (torch.nan_to_num(p - t, nan=0.0) ** 2) * valid.float()
            denom = valid.float().sum().clamp_min(1.0)
            mse_company = diff2.sum() / denom

            # IC company-level
            pN = torch.nan_to_num(p.squeeze(1).squeeze(-1), nan=0.0)  # [B,N]
            tN = torch.nan_to_num(t.squeeze(1).squeeze(-1), nan=0.0)  # [B,N]
            ic_per_b = _pearson_corr(pN, tN, dim=1)                    # [B]
            ic_company = ic_per_b.mean()

            total = mse_company + self.ic_weight * ((1.0 - ic_company) / 2.0)

            out["losses"] = {
                "mse": mse_company,
                "ic_company": ic_company,
                "total": total,
            }

        return out
