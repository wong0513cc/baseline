import torch
import torch.nn as nn
import math
from typing import List, Optional

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x):
        # x: [B, S, H]
        return self.fn(self.norm(x))

class MLP(nn.Module):
    def __init__(self, dim, hidden_mult=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim*hidden_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim*hidden_mult, dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class SwitchMultiModalBlock(nn.Module):
    """
    一層 Switch-Attention：
      - 輸入:  modal_list = [X1, X2, ..., XM]，每個 Xi 形狀 [B, K, N, H]
      - 作法:  第 l 層選擇某個模態 i 當 Q；其餘模態 concat 當 K/V
      - 輸出:  更新後的 modal_list，只有被選為 Q 的那個模態會被覆寫(殘差更新)
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.ffn = MLP(hidden_dim, hidden_mult=4, dropout=dropout)
        self.pre_attn = PreNorm(hidden_dim, nn.Identity())
        self.pre_ffn  = PreNorm(hidden_dim, nn.Identity())

    def _mhsa_sdpa(self, Q, K, V):
        B, SQ, H = Q.shape
        _, SK, _ = K.shape
        d = H // self.num_heads
        assert H % self.num_heads == 0

        inp_dtype = Q.dtype  # 記下進來時的 dtype（多半是 float32）

        #  所有 Linear + SDPA 都放在 autocast 內
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            Qh = self.q_proj(Q).view(B, SQ, self.num_heads, d).transpose(1, 2)
            Kh = self.k_proj(K).view(B, SK, self.num_heads, d).transpose(1, 2)
            Vh = self.v_proj(V).view(B, SK, self.num_heads, d).transpose(1, 2)

            out = torch.nn.functional.scaled_dot_product_attention(
                Qh, Kh, Vh, attn_mask=None, dropout_p=0.0, is_causal=False
            )  # [B, heads, SQ, d]

            out = out.transpose(1, 2).contiguous().view(B, SQ, H)
            out = self.out_proj(out)  # 線性也在 autocast 內

        # 回到呼叫者原本的 dtype（通常 fp32），避免後續層再出現 dtype mismatch
        return out.to(dtype=inp_dtype)

    def forward(self, modal_list: List[torch.Tensor], q_index: int, k_index: int, v_index: int):
        """
        modal_list: 長度 M 的 list，每個 [B,K,N,H]
        q_index/k_index/v_index: 本層指定的 Q/K/V 來源（pairwise）
        """
        B, K, N, H = modal_list[0].shape
        for x in modal_list:
            assert x.shape == (B, K, N, H)

        # 取出 Q/K/V 並展平成序列 S=K*N
        S = K * N
        Q = modal_list[q_index].reshape(B, S, H)  # [B,S,H]
        Kseq = modal_list[k_index].reshape(B, S, H)
        Vseq = modal_list[v_index].reshape(B, S, H)

        # PreNorm
        Qn  = self.pre_attn(Q)
        Kn  = self.pre_attn(Kseq)
        Vn  = self.pre_attn(Vseq)

        # Pairwise 注意力
        h = self._mhsa_sdpa(Qn, Kn, Vn)          # [B,S,H]

        # 殘差 + FFN
        Xq_new = Q + h
        Xq_new = Xq_new + self.ffn(self.pre_ffn(Xq_new))  # [B,S,H]
        Xq_new = Xq_new.view(B, K, N, H)                  # [B,K,N,H]

        out_list = []
        for m in range(len(modal_list)):
            out_list.append(Xq_new if m == q_index else modal_list[m])
        return out_list
    
class SwitchEncoder(nn.Module):
    def __init__(self, hidden_dim: int, depth: int = 6, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwitchMultiModalBlock(hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.depth = depth

        # 預設 pairwise 輪替（針對 4 模態）：(q,k,v)
        # 0: price, 1: fin, 2: event, 3: news  ← 請和你實際丟進來的順序保持一致
        self.schedule = [
            (0, 1, 1),  # 層0  Q=price  K/V=fin
            (1, 3, 3),  # 層1  Q=fin    K/V=news
            (3, 2, 2),  # 層2  Q=news   K/V=event
            (2, 0, 0),  # 層3  Q=event  K/V=price
        ]

    def forward(self, modal_list: List[torch.Tensor]):
        xs = modal_list
        M = len(xs)
        assert M == 4, "此 schedule 針對 4 模態；若 M != 4 請自訂 self.schedule"

        for l, block in enumerate(self.blocks):
            q_idx, k_idx, v_idx = self.schedule[l % len(self.schedule)]
            xs = block(xs, q_index=q_idx, k_index=k_idx, v_index=v_idx)

        fused = torch.stack(xs, dim=0).mean(dim=0)  # 也可以不用 fused，直接用 xs 各自池化
        return xs, fused