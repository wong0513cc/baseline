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
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors            # 新增：顏色轉換 HSV→RGB

from encoderSwitchAtt import ESGMultiModalModel
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
    log = {"loss_total":0.0,"mse":0.0,"ic_company":0.0,"steps":0}
    years = sorted(list(loaders_by_year.keys()))
    for y in years:
        for raw in loaders_by_year[y]:
            batch = move_inputs(raw, device, args.target)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                out = model(batch)  # forward expects dict
                losses = out.get("losses", None)
                if losses is None:
                    raise RuntimeError("Model did not return 'losses' dict; ensure label is provided and loss enabled.")
                loss = losses["total"]



            # 反傳前檢查 loss 是否有限
            if not torch.isfinite(loss):
                print(f"[WARN] loss not finite at epoch {epoch}: {float(loss)}")

            # 監看 AMP scale（出現 inf/NaN 會下降、甚至跳過 step）
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            if scale_after < scale_before:
                print(f"[AMP] scale decreased {scale_before} -> {scale_after} (possible inf/NaN grads)")


            log["loss_total"] += float(loss.detach().item())
            log["mse"]        += float(losses["mse"].detach().item())
            log["ic_company"]         += float(losses["ic_company"].detach().item())
            log["steps"]      += 1

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
    metrics = {"mse":0.0, "mae":0.0, "rmse":0.0, "smape":0.0, "ic_company": 0.0, "count":0}
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
                continue

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

def sizes_by_rank(values: np.ndarray, base: float = 6.0, max_extra: float = 10.0):
    """
    可選：依連續值調整點大小，例如依市值/ESG分位。
    values: 1D array。會做分位縮放；若不用就別呼叫。
    """
    if values is None or len(values) == 0:
        return None
    r = (values - values.min()) / (values.ptp() + 1e-8)
    return base + max_extra * r


def plot_modal_embeddings(model: nn.Module,
                          loaders_by_year: Dict[int, DataLoader],
                          device: torch.device,
                          args,
                          out_path: str,
                          year: int = None,
                          pool: str = "time",
                          max_points: int = 5000):
    """
    從某個 validation/test 年份抓一個 batch，取四模態 encoder 後的時間池化 [B,N,H]，
    攤平成公司集合（B*N, H），各模態各自做 PCA 2D，畫在 2x2 子圖。
    """
    model.eval()
    years = sorted(list(loaders_by_year.keys()))
    if not years:
        print("[plot_modal_embeddings] no years in loader.")
        return
    y = year if (year is not None and year in years) else years[0]

    # 取一個 batch
    raw = next(iter(loaders_by_year[y]))
    batch = move_inputs(raw, device, args.target)  # 轉到 [B,K,N,D] on device

    with torch.no_grad():
        enc = model.encode_modalities(batch, pool=pool)
        Pp = enc["pooled"]["price"]   # [B,N,H]
        Pf = enc["pooled"]["finance"]
        Pn = enc["pooled"]["news"]
        Pe = enc["pooled"]["event"]

        def _prep(X: torch.Tensor):
            X = X.detach().cpu().numpy()  # [B,N,H]
            X = X.reshape(-1, X.shape[-1])  # [B*N, H]
            if X.shape[0] > max_points:
                # 隨機下採樣避免點太多
                idx = np.random.choice(X.shape[0], max_points, replace=False)
                X = X[idx]
            return X

        Xp = _prep(Pp)
        Xf = _prep(Pf)
        Xn = _prep(Pn)
        Xe = _prep(Pe)

        def _pca2(x):
            if x.shape[0] < 3:
                # 點太少時，簡單補零
                z = np.zeros((x.shape[0], 2), dtype=np.float32)
            else:
                z = PCA(n_components=2).fit_transform(x)
            return z
        
        # 取一個 batch
        raw = next(iter(loaders_by_year[y]))
        batch = move_inputs(raw, device, args.target)  # 轉到 [B,K,N,D]

        # 取得公司清單（對齊 N），建議在這個可視化函式呼叫前，先用 batch_size=1
        symbols = raw.get("symbols", None)  # 你的 Dataset 若有提供，通常長度 = N；若沒有可略過上色

        Zp = _pca2(Xp); Zf = _pca2(Xf); Zn = _pca2(Xn); Ze = _pca2(Xe)

 # 準備對齊的顏色：只取第一個 batch（建議 B=1），N 個點
    color_map = None
    if isinstance(symbols, list):
        try:
            color_map = colors_by_symbol(symbols)  # shape [N, 3]
        except Exception:
            color_map = None

    # --- 繪圖 ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axlist = [axes[0,0], axes[0,1], axes[1,0], axes[1,1]]
    titles = ["Price (encoder)", "Finance (encoder)", "News (encoder)", "Event (encoder)"]
    data = [Zp, Zf, Zn, Ze]

    # 注意：這裡假設你在上面沒有對點做下採樣，且 B=1，則 Zp.shape[0] 應該 == N
    for ax, t, z in zip(axlist, titles, data):
        if z.shape[0] > 0:
            if (color_map is not None) and (len(color_map) == z.shape[0]):
                ax.scatter(z[:,0], z[:,1], s=8, c=color_map, alpha=0.9,
                           linewidths=0.2, edgecolors="k")
            else:
                ax.scatter(z[:,0], z[:,1], s=8, alpha=0.9, linewidths=0.2, edgecolors="k")
        ax.set_title(t)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(True, alpha=0.15)

    fig.suptitle(f"Modal Encoders Embeddings (year={y}, pool={pool})")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot_modal_embeddings] saved to {out_path}")


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
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--logdir", type=str, default="runs")
    ap.add_argument("--out_dir", type=str, default="./outputs")
    ap.add_argument("--predict_yearly", action="store_true", default=True)
    ap.add_argument("--years_train", type=str, default="2015,2016,2017,2018,2019,2020")
    ap.add_argument("--years_val", type=str, default="2021,2022")
    ap.add_argument("--years_test", type=str, default="2023,2024")
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
        ic_weight=0.0, ic_type="pearson"
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    history = {
        "train": {"loss_total": [], "mse": [], "ic_company": []},
        "val": {"mse": [], "mae": [], "rmse": [], "smape": [], "ic_company": []},
    }

    best_val = float("inf")
    best_path = os.path.join(args.out_dir, f"best_esg_{args.target}.pth")

    for epoch in range(1, args.epochs + 1):
        tr_log = train_one_epoch(model, optimizer, train_loaders, device, scaler, epoch, args)
        history["train"]["loss_total"].append(tr_log["loss_total"])
        history["train"]["mse"].append(tr_log["mse"])
        history["train"]["ic_company"].append(tr_log["ic_company"])

        # 驗證（不畫圖）
        val_metrics, val_detail = evaluate(model, val_loaders, device, args, desc="val")
        for k in ["mse","mae","rmse","smape","ic_company"]:
            history["val"].setdefault(k, []).append(val_metrics.get(k, float("nan")))


        # # 存這個 epoch 的 validation preds/labels
        val_csv_path = os.path.join(args.out_dir, f"val_preds_epoch{epoch:03d}_{args.target}.csv")
        _ = save_split_preds_labels(val_detail, out_path=val_csv_path, split="val", epoch=epoch)


        # 存最優
        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "val_mse": best_val}, best_path)

        plot_curves(history, os.path.join(args.out_dir, f"curves_{args.target}.png"))
        print(
            f"Epoch {epoch:03d} | Train total {tr_log['loss_total']:.4f} "
            f"(mse {tr_log['mse']:.4f}, ic_company {tr_log['ic_company']:.4f}) | "
            f"Val MSE {val_metrics['mse']:.4f} RMSE {val_metrics['rmse']:.4f} "
            f"IC_company {val_metrics.get('ic_company', float('nan')):.4f}"
            f"SSE {val_metrics.get('sse', float('nan')):.1f} "
            f"IC_company {val_metrics.get('ic_company', float('nan')):.4f}"
        )
        
    print(f"[INFO] best_val after training = {best_val:.6f}")
    # Load best and plot once for VAL
    if os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded best model from epoch {ckpt['epoch']} with val_mse={ckpt['val_mse']:.6f}")

    # 可視化：四模態 encoder embeddings（用驗證集第一個年份的一個 batch）
    plot_modal_embeddings(
        model, val_loaders, device, args,
        out_path=os.path.join(args.out_dir, f"modal_embeddings_{args.target}.png"),
        year=None,    # or years_val[0]
        pool="time",
        max_points=10**9  # 保證不下採樣，顏色才能對齊 N
    )

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
