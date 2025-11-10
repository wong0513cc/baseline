from typing import Dict, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Utils
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


# class BoundedHead(nn.Module):
#     def __init__(self, in_dim: int, lo: float = 0.0, hi: float = 1.0, dropout: float = 0.1, hidden: int = None):
#         super().__init__()
#         h = hidden or max(in_dim, 256)

#         # block1
#         self.ln1 = SafeLayerNorm(in_dim)
#         self.fc1 = nn.Linear(in_dim, h)
#         self.act = nn.ReLU()
#         self.drop = nn.Dropout(dropout)
#         self.proj1 = nn.Linear(h, in_dim)  # 投回原維度做殘差

#         # 輸出層
#         self.out = nn.Linear(in_dim, 1)

#         self.register_buffer("lo", torch.tensor(float(lo)))
#         self.register_buffer("hi", torch.tensor(float(hi)))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         y = self.proj1(self.drop(self.act(self.fc1(self.ln1(x)))))
#         x = x + y
#         z = self.out(x)
#         return torch.sigmoid(z) * (self.hi - self.lo) + self.lo
