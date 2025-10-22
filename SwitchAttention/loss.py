import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- 小投影頭：每個模態接一個 2 層 MLP，再做 L2 normalize ----
class Projector(nn.Module):
    def __init__(self, in_dim, hid=64, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(inplace=True),
            nn.Linear(hid, out_dim)
        )
    def forward(self, x):  # x: [..., D]
        z = self.net(x)
        return F.normalize(z, dim=-1)


# ---- 兩模態的雙向 InfoNCE，單月版本：把公司維 N 當 batch ----
def info_nce_two_modal_single_month(Zm, Zn, valid_mask=None, tau=0.07):
    """
    Zm, Zn: [N, d] (已 normalize)
    valid_mask: [N] bool，可為 None。False 會把該公司排除在此月的對比外。
    回傳：標量 loss
    """
    if valid_mask is not None:
        Zm = Zm[valid_mask]
        Zn = Zn[valid_mask]

    N = Zm.size(0)
    if N <= 1:   # 只剩 0/1 家公司就跳過
        return Zm.new_tensor(0.0)

    logits = (Zm @ Zn.t()) / tau             # [N, N]
    logits_t = logits.t()
    target = torch.arange(N, device=Zm.device)

    # 數值穩定（選配，但建議做）
    logits   = logits   - logits.max(dim=1, keepdim=True).values
    logits_t = logits_t - logits_t.max(dim=1, keepdim=True).values

    loss_m2n = F.cross_entropy(logits,   target)
    loss_n2m = F.cross_entropy(logits_t, target)
    return 0.5 * (loss_m2n + loss_n2m)

# ---- 六個模態配對，逐月平均 ----
def multimodal_icl_monthly(
    H_dict,                    # {'price': [B,K,N,Dp], 'fin': [B,K,N,Df], 'news': [B,K,N,Dn], 'event': [B,K,N,De]}
    projector_dict,            # {'price': Projector(Dp,...), ...}，各模態自己的投影頭
    valid_mask_dict=None,      # （可選）{'price':[B,K,N] bool, ...}；若無，傳 None
    tau=0.07
):
    """
    回傳：標量 loss（先對月份平均，再對 batch 平均，再對六個配對平均）
    假設 batch=1 也可正常運作。缺某模態時，從配對中自動跳過。
    """
    device = next(iter(projector_dict.values())).net[0].weight.device
    keys = [k for k,v in H_dict.items() if v is not None]   # 只取有提供的模態
    if len(keys) < 2:
        return torch.tensor(0.0, device=device)

    B, K = None, None
    for k in keys:
        assert H_dict[k].dim() == 4, f"{k} 期望 [B,K,N,D]"
        B, K = H_dict[k].shape[:2]
        break

    # 收集所有模態配對（六個配對）
    pairs = []
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            pairs.append((keys[i], keys[j]))
    if not pairs:
        return torch.tensor(0.0, device=device)

    losses_per_pair = []

    for m, n in pairs:
        Hm = H_dict[m]  # [B,K,N,Dm]
        Hn = H_dict[n]  # [B,K,N,Dn]
        Pm = projector_dict[m]
        Pn = projector_dict[n]

        # 可選的有效公司 mask：需同時存在於兩模態才參與
        Mmask = valid_mask_dict.get(m) if (valid_mask_dict is not None and m in valid_mask_dict) else None
        Nmask = valid_mask_dict.get(n) if (valid_mask_dict is not None and n in valid_mask_dict) else None

        # 對 batch 與月份逐一計算，再平均
        loss_months = []
        for b in range(B):
            for t in range(K):
                # 取出該月的 [N,D]
                Hm_bt = Hm[b, t]  # [N, Dm]
                Hn_bt = Hn[b, t]  # [N, Dn]

                # 投影 + normalize
                Zm_bt = Pm(Hm_bt)  # [N, d]
                Zn_bt = Pn(Hn_bt)  # [N, d]

                # 建立此月的有效公司 mask（若有）
                if (Mmask is not None) and (Nmask is not None):
                    vmask_bt = (Mmask[b, t] & Nmask[b, t]).to(torch.bool)  # [N]
                elif (Mmask is not None):
                    vmask_bt = Mmask[b, t].to(torch.bool)
                elif (Nmask is not None):
                    vmask_bt = Nmask[b, t].to(torch.bool)
                else:
                    vmask_bt = None

                loss_bt = info_nce_two_modal_single_month(Zm_bt, Zn_bt, vmask_bt, tau)
                loss_months.append(loss_bt)

        if len(loss_months) > 0:
            losses_per_pair.append(torch.stack(loss_months).mean())

    if len(losses_per_pair) == 0:
        return torch.tensor(0.0, device=device)

    return torch.stack(losses_per_pair).mean()


