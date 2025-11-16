# ===== baselines.py（可直接貼到 trainer.py 最後，或獨立成檔案 import） =====
import os, math, csv
import numpy as np
import torch

def _extract_label_company_from_label(label: torch.Tensor, target: str) -> torch.Tensor:
    """
    將 label 轉成 [B,1,N,1] 的 label_company：
    - 接受 [B,K,N,C] / [K,N,C] / [N,C] / [...,C]
    - 依 target 取通道（env/soc/gov/esg），再沿 K 取平均到 [B,1,N,1]
    """
    # 轉成 [B,K,N,C]
    if label.ndim == 2:         # [N,C]
        label = label.unsqueeze(0).unsqueeze(0)
    elif label.ndim == 3:       # [K,N,C]
        label = label.unsqueeze(0)
    elif label.ndim == 4:       # [B,K,N,C]
        pass
    else:
        raise ValueError(f"Unexpected label shape: {label.shape}")

    B,K,N,C = label.shape
    if target in {"env","soc","gov"}:
        ch = {"env":0, "soc":1, "gov":2}[target]
        lab = label[..., ch:ch+1] if C >= 1 else label[..., 0:1]  # [B,K,N,1]
    else:  # "esg" -> 取三通道平均；若 C=1 就直接用
        lab = label.mean(dim=-1, keepdim=True) if C > 1 else label[..., 0:1]  # [B,K,N,1]

    # 沿時間 K 聚合到 [B,1,N,1]
    lab_company = lab.mean(dim=1, keepdim=True)
    return lab_company

def _to_numpy_label_company(batch, target: str):
    """
    讀出公司層級標籤，輸出 y_np, mask_np（都是平坦化 [B*N]）
    來源優先順序：
      1) batch["label_company"]（若已存在）
      2) batch["label"] -> 轉成 label_company
    """
    if "label_company" in batch and batch["label_company"] is not None:
        y = batch["label_company"]  # [B,1,N,1]
        if y.ndim == 4:
            y = y.squeeze(1).squeeze(-1)  # [B,N]
        elif y.ndim == 3:
            y = y.squeeze(-1)             # [B,N]
        elif y.ndim == 2:
            pass
        else:
            raise ValueError(f"Unexpected label_company shape: {batch['label_company'].shape}")
    elif "label" in batch and batch["label"] is not None:
        lab_company = _extract_label_company_from_label(batch["label"], target)  # [B,1,N,1]
        y = lab_company.squeeze(1).squeeze(-1)  # [B,N]
    else:
        raise KeyError("Neither 'label_company' nor 'label' found in batch.")

    valid = ~torch.isnan(y)
    return y.detach().cpu().numpy(), valid.detach().cpu().numpy()


def _gather_split_labels(loaders_dict, target: str):
    by_year, by_year_mask, years_sorted = {}, {}, sorted(list(loaders_dict.keys()))
    for y in years_sorted:
        ys, ms = [], []
        for batch in loaders_dict[y]:
            y_np, m_np = _to_numpy_label_company(batch, target)  # << 帶 target
            ys.append(y_np.reshape(-1))
            ms.append(m_np.reshape(-1))
        by_year[y] = np.concatenate(ys, axis=0) if ys else np.array([])
        by_year_mask[y] = np.concatenate(ms, axis=0) if ms else np.array([])
    return by_year, by_year_mask, years_sorted

def _rmse(a, b, mask=None):
    if mask is not None:
        a = a[mask]; b = b[mask]
    if a.size == 0: return np.nan
    return math.sqrt(np.mean((a - b) ** 2))

def _mse(a, b, mask=None):
    if mask is not None:
        a = a[mask]; b = b[mask]
    if a.size == 0: return np.nan
    return np.mean((a - b) ** 2)

def _mae(a, b, mask=None):
    if mask is not None:
        a = a[mask]; b = b[mask]
    if a.size == 0: return np.nan
    return np.mean(np.abs(a - b))

def _write_csv(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    header = ["split","baseline","year","count","rmse","mse","mae","true_std","pred_std","note"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r.get(k,"") for k in header])

def compute_and_save_baselines(train_loaders, val_loaders, test_loaders, out_dir, split_names=("val","test"), target: str="esg"):
    """
    會輸出:
      - {out_dir}/baselines_val.csv
      - {out_dir}/baselines_test.csv
    內容包含每年與 Overall 的 RMSE/MSE/MAE，以及 true/pred 的 std。
    """
    rows_out = []
    for split in split_names:
        loaders = val_loaders if split == "val" else test_loaders
        rows = []

        # 1) 收集各年 label 與有效 mask
        by_year, by_year_mask, years_sorted = _gather_split_labels(loaders, target)

        # 便利：把各年拼成全域向量
        all_y = np.concatenate([by_year[y][by_year_mask[y]] for y in years_sorted if by_year[y].size > 0]) \
                if years_sorted else np.array([])
        if all_y.size == 0:
            _write_csv(rows, os.path.join(out_dir, f"baselines_{split}.csv"))
            continue

        # ========== Baseline A：全域均值 ==========
        mu_global = np.mean(all_y)
        # per-year
        for y in years_sorted:
            yv = by_year[y]; mv = by_year_mask[y]
            pred = np.full_like(yv, fill_value=mu_global, dtype=float)
            rows.append({
                "split": split, "baseline": "global_mean", "year": y,
                "count": int(mv.sum()),
                "rmse": _rmse(yv, pred, mv), "mse": _mse(yv, pred, mv), "mae": _mae(yv, pred, mv),
                "true_std": np.std(yv[mv]) if mv.any() else np.nan,
                "pred_std": np.std(pred[mv]) if mv.any() else np.nan,
                "note": "predict overall mean"
            })
        # overall
        pred_all = np.full_like(all_y, fill_value=mu_global, dtype=float)
        rows.append({
            "split": split, "baseline": "global_mean", "year": "OVERALL",
            "count": int(all_y.size),
            "rmse": _rmse(all_y, pred_all), "mse": _mse(all_y, pred_all), "mae": _mae(all_y, pred_all),
            "true_std": float(np.std(all_y)),
            "pred_std": float(np.std(pred_all)),
            "note": "predict overall mean"
        })

        # ========== Baseline B：逐年均值 ==========
        # per-year
        preds_concat, trues_concat = [], []
        for y in years_sorted:
            yv = by_year[y]; mv = by_year_mask[y]
            if yv.size == 0: continue
            mu_y = np.mean(yv[mv]) if mv.any() else np.mean(yv)
            pred = np.full_like(yv, fill_value=mu_y, dtype=float)
            rows.append({
                "split": split, "baseline": "per_year_mean", "year": y,
                "count": int(mv.sum()),
                "rmse": _rmse(yv, pred, mv), "mse": _mse(yv, pred, mv), "mae": _mae(yv, pred, mv),
                "true_std": np.std(yv[mv]) if mv.any() else np.nan,
                "pred_std": np.std(pred[mv]) if mv.any() else np.nan,
                "note": "predict mean of the same year"
            })
            preds_concat.append(pred[mv]); trues_concat.append(yv[mv])
        # overall（把各年有效樣本串起來衡量）
        if preds_concat:
            P = np.concatenate(preds_concat); T = np.concatenate(trues_concat)
            rows.append({
                "split": split, "baseline": "per_year_mean", "year": "OVERALL",
                "count": int(T.size),
                "rmse": _rmse(T, P), "mse": _mse(T, P), "mae": _mae(T, P),
                "true_std": float(np.std(T)),
                "pred_std": float(np.std(P)),
                "note": "predict mean of each year"
            })

        # ========== Baseline C：上一年標籤（lag-1） ==========
        # 假設公司索引在同一 split 內「跨年一致」；否則需用 symbols 做對齊
        preds_concat, trues_concat = [], []
        for i, y in enumerate(years_sorted):
            if i == 0:  # 第一個年沒有上一年
                rows.append({
                    "split": split, "baseline": "lag1", "year": y, "count": 0,
                    "rmse": np.nan, "mse": np.nan, "mae": np.nan,
                    "true_std": np.nan, "pred_std": np.nan,
                    "note": "no previous year"
                })
                continue
            y_prev = years_sorted[i-1]
            yv = by_year[y]; mv = by_year_mask[y]
            yv_prev = by_year[y_prev]; mv_prev = by_year_mask[y_prev]

            # 只在「兩年都有效」的位置做對齊
            M = min(yv.size, yv_prev.size)
            if M == 0:
                rows.append({
                    "split": split, "baseline": "lag1", "year": y, "count": 0,
                    "rmse": np.nan, "mse": np.nan, "mae": np.nan,
                    "true_std": np.nan, "pred_std": np.nan,
                    "note": "no overlapping size"
                })
                continue

            mv2 = mv[:M] & mv_prev[:M]
            t = yv[:M]; p = yv_prev[:M]
            rows.append({
                "split": split, "baseline": "lag1", "year": y,
                "count": int(mv2.sum()),
                "rmse": _rmse(t, p, mv2), "mse": _mse(t, p, mv2), "mae": _mae(t, p, mv2),
                "true_std": np.std(t[mv2]) if mv2.any() else np.nan,
                "pred_std": np.std(p[mv2]) if mv2.any() else np.nan,
                "note": f"predict year {y_prev} label for year {y}"
            })
            preds_concat.append(p[mv2]); trues_concat.append(t[mv2])

        if preds_concat:
            P = np.concatenate(preds_concat); T = np.concatenate(trues_concat)
            rows.append({
                "split": split, "baseline": "lag1", "year": "OVERALL",
                "count": int(T.size),
                "rmse": _rmse(T, P), "mse": _mse(T, P), "mae": _mae(T, P),
                "true_std": float(np.std(T)),
                "pred_std": float(np.std(P)),
                "note": "predict previous year label"
            })
        rows_out.append((split, rows))

        # 輸出 CSV
        for split, rows in rows_out:
            out_csv = os.path.join(out_dir, f"baselines_{split}_{target}.csv")
            _write_csv(rows, out_csv)
            print(f"[BASELINE] saved -> {out_csv}")
