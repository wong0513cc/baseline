import os
import math
import json
import argparse
import random
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors            # 新增：顏色轉換 HSV→RGB

from model.encoderClSwitchAtt import ESGMultiModalModel
from dataset_v2 import GraphESGDataset
from dataloader import build_loaders

TARGET2IDX = {"env": 0, "soc": 1, "gov": 2}

# -------------------------------
# Utils
# -------------------------------

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
# -------------------------------
# Train/Eval
# -------------------------------

def move_inputs(batch: dict, device: torch.device, target: str):
    to = lambda t: (t.float().to(device) if isinstance(t, torch.Tensor) else t)
    price   = to(batch["price"]);   finance = to(batch["finance"])
    event   = to(batch["event"]);   news    = to(batch["news"])
    if price.ndim == 3: price = price.unsqueeze(0)
    if finance.ndim == 3: finance = finance.unsqueeze(0)
    if event.ndim == 3: event = event.unsqueeze(0)
    if news.ndim == 3: news = news.unsqueeze(0)

    bd = {"price":price, "finance":finance, "news":news, "event":event}
    if "label" in batch and batch["label"] is not None:
        lab = to(batch["label"])
        lab_company= select_labels_company_and_overall(lab, target)
        bd["label_company"] = lab_company.to(device)  # [B,1,N,1]
    return bd


def train_one_epoch(model: nn.Module,
                    optimizer: optim.Optimizer,
                    loaders_by_year: Dict[int, DataLoader],
                    device: torch.device,
                    scaler: torch.cuda.amp.GradScaler,
                    epoch: int,
                    args) -> Dict[str, float]:

    model.train()

    #### debug 1 ：訓練前觀察 head 權重大小
    with torch.no_grad():
        head_w_before = 0.0
        for n, p in model.named_parameters():
            if p.requires_grad and "company_head" in n:
                head_w_before += p.norm().item()
    print(f"[DBG#1][epoch {epoch}] head ||w|| BEFORE =", head_w_before)

    log = {"loss_total":0.0,"mse":0.0,"ic_company":0.0,"icl":0.0,"steps":0}
    

    years = sorted(list(loaders_by_year.keys()))
    for y in years:

        step = 0
        for raw in loaders_by_year[y]:
            batch = move_inputs(raw, device, args.target)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                out = model(batch)
                losses = out.get("losses", None)
                if losses is None:
                    raise RuntimeError("Model did not return 'losses' dict; ensure label is provided and loss enabled.")
                loss = losses["total"]
                    

            # 反傳與更新
            if not torch.isfinite(loss):
                print(f"[WARN] loss not finite at epoch {epoch}: {float(loss)}")

            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()

            # #### DBG 2：反傳「後」看梯度是否流動
            # total_grad = 0.0
            # cnt = 0
            # for n,p in model.named_parameters():
            #     if p.grad is not None:
            #         total_grad += float(p.grad.detach().abs().mean().item())
            #         cnt += 1
            # print(f"[DBG#2] mean|grad| over params: {total_grad/max(1,cnt):.3e}")

            if args.grad_clip is not None and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            if scale_after < scale_before:
                print(f"[AMP] scale decreased {scale_before} -> {scale_after} (possible inf/NaN grads)")

            #  # ===== DBG 3：這個 batch 更新後，head 權重是否有改變 =====
            # with torch.no_grad():
            #     head_w_after_batch = 0.0
            #     for n,p in model.named_parameters():
            #         if p.requires_grad and "company_head" in n:
            #             head_w_after_batch += p.norm().item()
            # print(f"[DBG#3][epoch {epoch} y{y} step{step}] head ||w|| batch-UPDATED =", head_w_after_batch,
            #       f"(amp scale {scale_before:.1e}->{scale_after:.1e})")
            # step += 1

        

            # logging
            log["loss_total"] += float(loss.detach().item())
            log["mse"]        += float(losses["mse"].detach().item())
            log["ic_company"] += float(losses["ic_company"].detach().item())
            log["icl"] += float(losses.get("icl", 0.0))
            log["steps"]      += 1

    # ===== epoch 結束：印 head 的相對更新量 =====
    with torch.no_grad():
        dw2, w2 = 0.0, 0.0
        for n, p in model.named_parameters():
            if p.requires_grad and "company_head" in n:
                if hasattr(p, "_prev"):
                    dw2 += torch.sum((p - p._prev)**2).item()
                    w2  += torch.sum(p**2).item()
                # 更新快照（下個 epoch 用）
                p._prev = p.detach().clone()
        if w2 > 0:
            print(f"[DBG2] head rel_update (epoch {epoch}) = {math.sqrt(dw2)/math.sqrt(w2):.3e}")

    for k in list(log.keys()):
        if k != "steps":
            log[k] = log[k] / max(1, log["steps"])
    return log


        
@torch.no_grad()
def evaluate(model: nn.Module,
             loaders_by_year: Dict[int, DataLoader],
             device: torch.device,
             args,
             desc="val") -> Tuple[Dict[str, float], Dict[int, dict]]:
    """
    評估使用「公司層級」：
      - 取 pred_company [B,1,N,1] 與 label_company [B,1,N,1]
      - 將 B 與 N 攤平成一維，對所有公司計算 mse/mae/rmse/smape
      - ic_company 也沿公司維度計算（pearson over all companies across years）
    per_year[y] 會存該年的「公司層級」向量（preds/labels 長度 ~ N * B）
    """
    model.eval()
    metrics = {"mse":0.0, "mae":0.0, "rmse":0.0, "smape":0.0, "ic_company": 0.0, "icl": 0.0, "count":0}
    per_year = {}

    all_preds_company, all_labels_company = [], []

    years = sorted(list(loaders_by_year.keys()))
    for y in years:
        preds_y_list, labels_y_list = [], []
        symbols_y: List[str] = []

        for raw in loaders_by_year[y]:
            batch = move_inputs(raw, device, args.target)
            out = model(batch)

            # 需要公司層級標籤與輸出
            if ("label_company" not in batch) or ("pred_company" not in out):
                break

            pc = out["pred_company"].squeeze(1).squeeze(-1).detach().cpu().numpy()  # [B,N]
            lc = batch["label_company"].squeeze(1).squeeze(-1).detach().cpu().numpy()  # [B,N]

            # 攤平成向量（B*N）
            preds_y_list.append(pc.reshape(-1))
            labels_y_list.append(lc.reshape(-1))

            # 盡力從 raw 取 symbols（batch_size=1 時最準確）
            syms = raw.get("symbols", None)
            if syms is not None and len(symbols_y) == 0:
                # 若 B>1，這裡不一定能拿到全部 symbols；先保守處理
                if isinstance(syms, list):
                    symbols_y = syms
                else:
                    try:
                        symbols_y = list(syms)
                    except Exception:
                        pass

        if len(preds_y_list) == 0:
            continue

        preds_y = np.concatenate(preds_y_list)   # [~B*N]
        labels_y = np.concatenate(labels_y_list)  # [~B*N]

        se_y  = (labels_y - preds_y) ** 2
        sse_y = float(se_y.sum())                # 加總（你要的）
        mse_y = float(se_y.mean()) 

        mae_y = float(np.abs(labels_y - preds_y).mean())
        rmse_y = math.sqrt(mse_y)
        smape_y = smape(labels_y, preds_y, eps=1e-8, percent=True)
        ic_company_y = pearsonr_safe(labels_y, preds_y)

        # 記錄年度明細
        # symbols 對齊長度：若 symbols_y 長度與公司數不對，就省略 symbols
        per_year[y] = {
            "sse": sse_y, "mse": mse_y,
            "mae": mae_y,
            "rmse": rmse_y,
            "smape": smape_y,
            "ic_company": ic_company_y,
            "preds": preds_y, "labels": labels_y,
            "symbols": symbols_y if len(symbols_y) == preds_y.size else None
        }

        all_preds_company.append(preds_y)
        all_labels_company.append(labels_y)

    # 匯總（所有年份 × 公司）
    if len(all_preds_company) > 0:
        all_preds_company = np.concatenate(all_preds_company)
        all_labels_company = np.concatenate(all_labels_company)

        mse = float(((all_labels_company - all_preds_company)**2).mean())
        mae = float(np.abs(all_labels_company - all_preds_company).mean())
        rmse = math.sqrt(mse)
        smape_all = smape(all_labels_company, all_preds_company, eps=1e-8, percent=True)
        ic_company = pearsonr_safe(all_labels_company, all_preds_company)

        metrics.update({"mse":mse,"mae":mae,"rmse":rmse,"smape":smape_all,
                        "ic_company": ic_company, "count": all_preds_company.size})
        

            # ===== 這裡加：基線與分佈對照 =====
        Yh = all_preds_company
        Y  = all_labels_company
        rmse_model = float(np.sqrt(np.mean((Y - Yh)**2)))
        rmse_mean  = float(np.sqrt(np.mean((Y - Y.mean())**2)))
        print(f"[DBG2][{desc}] RMSE(model)={rmse_model:.4f} vs RMSE(mean)={rmse_mean:.4f}")
        print(f"[DBG3][{desc}] pred std={float(Yh.std()):.4f}  true std={float(Y.std()):.4f}")

    return metrics, per_year


def save_split_preds_labels(per_year: Dict[int, dict], out_path: str, split: str = "val", epoch: int = None):
    """
    把 evaluate 傳回的 per_year 中的 preds/labels 存成一個 CSV。
    欄位：split, epoch, year, symbol, idx, pred, label
    - symbol 若拿不到則用 IDXi
    - idx 是該年的公司索引（0..N-1）
    """
    rows = []
    for y, d in sorted(per_year.items()):
        preds = np.asarray(d["preds"])
        labels = np.asarray(d["labels"])
        symbols = d.get("symbols", None)
        n = len(preds)
        for i in range(n):
            sym = (symbols[i] if (symbols is not None and i < len(symbols)) else f"IDX{i}")
            rows.append({
                "split": split,
                "epoch": (int(epoch) if epoch is not None else None),
                "year": int(y),
                "symbol": sym,
                "idx": int(i),
                "pred": float(preds[i]),
                "label": float(labels[i]),
            })
    df = pd.DataFrame(rows, columns=["split","epoch","year","symbol","idx","pred","label"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    return df

# 畫Curves
def plot_curves(history: dict, out_dir: str):
    epochs = np.arange(1, len(history.get("train", {}).get("loss_total", [])) + 1)
    if len(epochs) == 0:
        return

    # 取資料（安全取用，避免 KeyError）
    tr_total = history.get("train", {}).get("loss_total", [])
    val_mse  = history.get("val",   {}).get("mse", [])
    tr_ic    = history.get("train", {}).get("ic_company", [])
    val_ic   = history.get("val",   {}).get("ic_company", [])

    fig, ax1 = plt.subplots()

    # 左軸：Loss / MSE
    ax1.plot(epochs, tr_total, label="train_total", linewidth=2)
    if len(val_mse) == len(epochs) and len(val_mse) > 0:
        ax1.plot(epochs, val_mse, label="val_mse", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss / MSE")
    ax1.set_title("Training & Validation")
    ax1.grid(True, alpha=0.25)

    # 右軸：IC（公司層級）
    ax2 = ax1.twinx()
    has_any_ic = False
    if len(tr_ic) == len(epochs) and len(tr_ic) > 0:
        ax2.plot(epochs, tr_ic, "--", label="train_ic_company", linewidth=2)
        has_any_ic = True
    if len(val_ic) == len(epochs) and len(val_ic) > 0:
        ax2.plot(epochs, val_ic, "--", label="val_ic_company", linewidth=2)
        has_any_ic = True
    if has_any_ic:
        ax2.set_ylabel("IC (company-level)")

    # 合併圖例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    fig.tight_layout()
    fig.savefig(out_dir, dpi=150)
    plt.close(fig)

# 畫散佈圖
def plot_scatter(pred: np.ndarray, label: np.ndarray, title: str, out_path: str):
    plt.figure()
    plt.scatter(label, pred, s=8, alpha=0.6)
    lo = float(min(label.min(), pred.min()))
    hi = float(max(label.max(), pred.max()))
    plt.plot([lo,hi], [lo,hi], linestyle='--')
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_scatter_from_year_detail(per_year: Dict[int, dict], title: str, out_path: str):
    if len(per_year) == 0:
        return
    years_sorted = sorted(per_year.keys())
    preds = np.concatenate([per_year[y]["preds"] for y in years_sorted])
    labels = np.concatenate([per_year[y]["labels"] for y in years_sorted])
    plot_scatter(preds, labels, title, out_path)

def colors_by_symbol(symbols: List[str]):
    """
    給每個 symbol 一個固定顏色（跨執行仍一致）。
    回傳 numpy array shape [len(symbols), 3] 的 RGB。
    """
    def _stable_hash(s: str) -> int:
        # FNV-1a 32-bit
        h = 2166136261
        for ch in s.encode("utf-8"):
            h = (h ^ ch) * 16777619
        return h & 0xffffffff

    hues = np.array([(_stable_hash(str(s)) % 360) / 360.0 for s in symbols], dtype=float)
    sat, val = 0.65, 0.85
    colors = [mcolors.hsv_to_rgb((h, sat, val)) for h in hues]
    return np.array(colors)


def save_test_csv(per_year: Dict[int, dict], out_path: str):
    rows = []
    for y, d in per_year.items():
        preds = d["preds"]
        labels = d["labels"]
        symbols = d.get("symbols", None)
        for i in range(len(preds)):
            sym = (symbols[i] if (symbols is not None and i < len(symbols)) else f"IDX{i}")
            rows.append({"year": y, "symbol": sym, "pred": float(preds[i]), "label": float(labels[i])})
    df = pd.DataFrame(rows, columns=["year","symbol","pred","label"])
    df.to_csv(out_path, index=False)
    return df



def save_test_year_metrics(per_year: Dict[int, dict], years: List[int], out_path: str):
    """
    依 per_year（evaluate 回傳）萃取指定年份的指標，存成 CSV。
    欄位：year, mse, rmse, mae, smape, ic
    若 per_year[y] 沒有某指標，就用 preds/labels 現算補上。
    """
    rows = []
    for y in years:
        d = per_year.get(y)
        if d is None:
            continue

        # 先嘗試拿 evaluate 算好的值
        mse   = d.get("mse",   None)
        mae   = d.get("mae",   None)
        rmse  = d.get("rmse",  None)
        smape = d.get("smape", None)
        ic    = d.get("ic_company", None)

        rows.append({
            "year": int(y),
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "smape": float(smape),
            "ic": float(ic),
        })

    df = pd.DataFrame(rows, columns=["year","mse","rmse","mae","smape","ic"])
    df.to_csv(out_path, index=False)
    return df

# -------------------------------
# Main
# -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=str, default="esg", choices=["env","soc","gov","esg"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda:1")
    ap.add_argument("--logdir", type=str, default="runs")
    ap.add_argument("--out_dir", type=str, default="./outputs")
    ap.add_argument("--predict_yearly", action="store_true", default=True)
    ap.add_argument("--years_train", type=str, default="2015,2016,2017,2018,2019,2020")
    ap.add_argument("--years_val", type=str, default="2021,2022")
    ap.add_argument("--years_test", type=str, default="2023,2024")

    # early stopping
    ap.add_argument("--early_stop_patience", type=int, default=10)   # 連續幾個 epoch 沒進步就停
    ap.add_argument("--early_stop_min_delta", type=float, default=0) # 最小改善幅度（例如 1e-4

    # data roots
    ap.add_argument("--root_price", type=str, required=True)
    ap.add_argument("--root_finance", type=str, required=True)
    ap.add_argument("--root_news", type=str, required=True)
    ap.add_argument("--root_event", type=str, required=True)
    ap.add_argument("--root_graph", type=str, required=True)
    ap.add_argument("--root_label", type=str, required=True)
    ap.add_argument("--root_year_symbols", type=str, required=True)

    args = ap.parse_args()

    # set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    root_paths = {
        "price": args.root_price,
        "finance": args.root_finance,
        "news": args.root_news,
        "event": args.root_event,
        "graph": args.root_graph,
        "label": args.root_label,
        "year_symbols": args.root_year_symbols,
    }

    years_train = [int(x) for x in args.years_train.split(",") if x.strip()]
    years_val   = [int(x) for x in args.years_val.split(",") if x.strip()]
    years_test  = [int(x) for x in args.years_test.split(",") if x.strip()]

    # loaders
    train_loaders, val_loaders, test_loaders = build_loaders(
        years_train, years_val, years_test, args.batch_size, root_paths
    )

    # 先從一個 batch 推斷各模態維度，建立模型
    sample_year = years_train[0]
    sample_batch = next(iter(train_loaders[sample_year]))
    def _infer_dim(x):
        t = x if isinstance(x, torch.Tensor) else torch.tensor(x)
        if t.ndim == 3:  # [K,N,D]
            return t.shape[-1]
        elif t.ndim == 4:  # [B,K,N,D]
            return t.shape[-1]
        else:
            raise ValueError(f"Unexpected tensor ndim={t.ndim} for inferring D.")
    Dp = _infer_dim(sample_batch["price"])
    Df = _infer_dim(sample_batch["finance"])
    Dn = _infer_dim(sample_batch["news"])
    De = _infer_dim(sample_batch["event"])

    model = ESGMultiModalModel(
        d_price=Dp, d_finance=Df, d_news=Dn, d_event=De,
        hidden=64,
        lstm_layers=1, lstm_bidirectional=False,
        dropout=0.1,
        nhead_time=4,
        news_layers=2, event_layers=2,
        ic_weight=0.1, ic_type="pearson"
    ).to(device)

    def build_param_groups(model, lr=1e-3, wd=1e-4):
        decay, nodecay = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            n_lower = n.lower()
            if n_lower.endswith("bias") or "layernorm" in n_lower or ".ln" in n_lower:
                nodecay.append(p)   # no weight decay
            else:
                decay.append(p)
        return [
            {"params": decay,   "weight_decay": wd,  "lr": lr},
            {"params": nodecay, "weight_decay": 0.0, "lr": lr},
        ]

    optimizer = optim.AdamW(build_param_groups(model), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    history = {
        "train": {"loss_total": [], "mse": [], "ic_company": [], "icl": [],},
        "val": {"mse": [], "mae": [], "rmse": [], "smape": [], "ic_company": []},
    }

    best_val = float("inf")
    best_path = os.path.join(args.out_dir, f"best_esg_{args.target}.pth")
    no_improve = 0
    best_epoch = None

    for epoch in range(1, args.epochs + 1):
        tr_log = train_one_epoch(model, optimizer, train_loaders, device, scaler, epoch, args)
        history["train"]["loss_total"].append(tr_log["loss_total"])
        history["train"]["mse"].append(tr_log["mse"])
        history["train"]["ic_company"].append(tr_log["ic_company"])
        history["train"]["icl"].append(tr_log["icl"])

        # 驗證（不畫圖）
        val_metrics, val_detail = evaluate(model, val_loaders, device, args, desc="val")
        for k in ["mse","mae","rmse","smape","ic_company"]:
            history["val"].setdefault(k, []).append(val_metrics.get(k, float("nan")))

        # 以 val MSE 當監控指標（也可改成 RMSE）
        cur = val_metrics["mse"]
        improved = (best_val - cur) > args.early_stop_min_delta

        if improved:
            best_val = cur
            best_epoch = epoch
            no_improve = 0
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "val_mse": best_val}, best_path)
        else:
            no_improve += 1

        # 畫曲線 & 印 log（保持不變）
        plot_curves(history, os.path.join(args.out_dir, f"curves_{args.target}.png"))
        print(
            f"Epoch {epoch:03d} | Train total {tr_log['loss_total']:.4f} "
            f"(mse {tr_log['mse']:.4f}, ic_company {tr_log['ic_company']:.4f}), icl_loss {tr_log['icl']} | "
            f"Val MSE {val_metrics['mse']:.4f} RMSE {val_metrics['rmse']:.4f} "
            f"ic_company {val_metrics.get('ic_company', float('nan')):.4f} | "
            f"no_improve={no_improve}/{args.early_stop_patience}"
        )

        # 觸發 Early Stop
        if no_improve >= args.early_stop_patience:
            print(f"[EARLY STOP] no improvement for {args.early_stop_patience} epochs "
                f"(best epoch {best_epoch}, best val_mse={best_val:.6f}).")
            break

        # # 存最優
        # if val_metrics["mse"] < best_val:
        #     best_val = val_metrics["mse"]
        #     torch.save({"epoch": epoch, "state_dict": model.state_dict(), "val_mse": best_val}, best_path)

        # plot_curves(history, os.path.join(args.out_dir, f"curves_{args.target}.png"))
        # print(
        #     f"Epoch {epoch:03d} | Train total {tr_log['loss_total']:.4f} "
        #     f"(mse {tr_log['mse']:.4f}, ic_company {tr_log['ic_company']:.4f}), icl_loss {tr_log['icl']} | "
        #     f"Val MSE {val_metrics['mse']:.4f} RMSE {val_metrics['rmse']:.4f} "
        #     f"IC_company {val_metrics.get('ic_company', float('nan')):.4f}"
        # )
        
    print(f"[INFO] best_val after training = {best_val:.6f}")
    # Load best and plot once for VAL
    if os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded best model from epoch {ckpt['epoch']} with val_mse={ckpt['val_mse']:.6f}")

    val_metrics, val_detail = evaluate(model, val_loaders, device, args, desc="val(best)")
    if len(val_detail) > 0:
        plot_scatter_from_year_detail(
            val_detail,
            title=f"Validation Scatter (best)",
            out_path=os.path.join(args.out_dir, f"val_scatter_{args.target}.png")
        )

    val_csv_path = os.path.join(args.out_dir, f"val_results_{args.target}.csv")
    _ = save_test_csv(val_detail, val_csv_path)
    print(f"Saved test CSV to {val_csv_path}")

    # TEST once
    test_metrics, test_detail = evaluate(model, test_loaders, device, args, desc="test")
    print("Test:", test_metrics)
    if len(test_detail) > 0:
        plot_scatter_from_year_detail(
            test_detail,
            title=f"Test Scatter (best)",
            out_path=os.path.join(args.out_dir, f"test_scatter_{args.target}.png")
        )

    csv_path = os.path.join(args.out_dir, f"test_results_{args.target}.csv")
    _ = save_test_csv(test_detail, csv_path)
    print(f"Saved test CSV to {csv_path}")

    year_csv_path = os.path.join(args.out_dir, f"test_year_metrics_{args.target}.csv")
    _ = save_test_year_metrics(test_detail, years=[2023, 2024], out_path=year_csv_path)
    print(f"Saved per-year test metrics CSV to {year_csv_path}")

    with open(os.path.join(args.out_dir, f"history_{args.target}.json"), "w") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()
