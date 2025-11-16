from typing import Dict, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np


# model
class SafeLayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=1e5, neginf=-1e5)
        return super().forward(x)

class PositionalEncoding(nn.Module):
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
        

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1], :]
        
class head(nn.Module):
    def __init__(self, in_dim: int, lo: float = 0.0, hi: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            # SafeLayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, 1)
        )
        self.register_buffer("lo", torch.tensor(float(lo)))
        self.register_buffer("hi", torch.tensor(float(hi)))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return torch.sigmoid(z) * (self.hi - self.lo) + self.lo
    
#trainer

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def smape(y_true, y_pred, eps=1e-8, percent=True):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    num = 2.0 * np.abs(y_pred - y_true)
    den = np.abs(y_true) + np.abs(y_pred) + eps
    val = np.mean(num / den)
    return val * 100.0 if percent else val

def pearsonr_safe(x, y):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.size < 2 or y.size < 2:
        return np.nan
    xm = x - x.mean()
    ym = y - y.mean()
    denom = (np.linalg.norm(xm) * np.linalg.norm(ym))
    if denom == 0:
        return np.nan
    return float(np.dot(xm, ym) / denom)

def select_labels_company_and_overall(label: torch.Tensor, target: str):
    """
    回傳：
      - label_company: [B,1,N,1]（年度、逐公司）
    """
    # 整到 [B,K,N,C]
    if label.ndim == 2:       # [N,C]
        label = label.unsqueeze(0).unsqueeze(0)
    elif label.ndim == 3:     # [K,N,C]
        label = label.unsqueeze(0)
    elif label.ndim == 4:     # [B,K,N,C]
        pass
    else:
        raise ValueError(f"Unexpected label shape: {label.shape}")

    B, K, N, C = label.shape
    # 取通道
    if target in {"env","soc","gov"}:
        ch = {"env":0,"soc":1,"gov":2}[target]
        lab = label[..., ch:ch+1] if C == 3 else label[..., 0:1]  # [B,K,N,1]
    else:
        lab = label.mean(dim=-1, keepdim=True) if C == 3 else label[..., 0:1]

    # 年度聚合
    lab_company = lab.mean(dim=1, keepdim=True)  # [B,1,N,1]（沿 K）
    return lab_company