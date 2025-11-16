import os
import math
import json
import argparse
import random
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors


# draw + csv

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
