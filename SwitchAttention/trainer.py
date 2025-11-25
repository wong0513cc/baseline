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
import matplotlib.colors as mcolors
from utils import pearsonr_safe, smape, select_labels_company_and_overall, set_seed
from draw import save_split_preds_labels, plot_curves, plot_scatter, plot_scatter_from_year_detail, save_test_csv, save_test_year_metrics
from model.encoderClSwitchAtt import ESGMultiModalModel
from dataset import GraphESGDataset
from dataloader import build_loaders
from baseline import _gather_split_labels, compute_and_save_baselines, _to_numpy_label_company, _rmse, _mse, _mae, _write_csv, _extract_label_company_from_label

TARGET2IDX = {"env": 0, "soc": 1, "gov": 2}

def plot_temporal_attention_heatmap(alphas_dict: Dict[str, List[np.ndarray]], out_path: str, title: str):
    """
    繪製 4x12 的時間注意力熱圖 (支援不同 N 的批次)
    
    Args:
        alphas_dict (dict): 
            key 是模態名稱 (e.g., "price"), 
            value 是 numpy 陣列的 *列表*
            例如: {"price": [array([B1, N1, 12]), array([B2, N2, 12])], ...}
        out_path (str): 圖片儲存路徑
        title (str): 圖片標題
    """
    modalities = ["price", "fin", "news", "event"]
    if not all(m in alphas_dict for m in modalities):
        print(f"Warning: Not all modalities found. Skipping plot.")
        return

    K = alphas_dict["price"][0].shape[-1] # 獲取時間步長 (e.g., 12)
    
    # 1. 計算 "加權" 平均注意力
    avg_alphas = []
    for m in modalities:
        batch_alphas_list = alphas_dict[m] # 這是 [array([B1,N1,K]), array([B2,N2,K]), ...]
        
        total_weighted_sum = np.zeros(K)
        total_samples = 0
        
        for batch_alpha_array in batch_alphas_list:
            # batch_alpha_array is [B, N, K] (例如 [1, 1066, 12])
            
            # 總樣本數 = B * N
            num_samples_in_batch = batch_alpha_array.shape[0] * batch_alpha_array.shape[1]
            
            # (B, N, K) -> (K,) 
            # 先計算這個 batch (例如 B=1, N=1066) 的平均
            mean_alpha_this_batch = np.mean(batch_alpha_array, axis=(0, 1)) # [K,]
            
            # 累加 (平均值 * 樣本數)
            total_weighted_sum += (mean_alpha_this_batch * num_samples_in_batch)
            total_samples += num_samples_in_batch
        
        # 最終的加權平均值
        if total_samples == 0:
            avg_alpha_m = np.zeros(K) # 避免除以 0
        else:
            avg_alpha_m = total_weighted_sum / total_samples # [K,]
        
        avg_alphas.append(avg_alpha_m)
        
    # 2. 堆疊成 (4, 12) 的矩陣
    heatmap_data = np.stack(avg_alphas, axis=0) # [4, K]
    
    # 3. 繪圖 (這部分程式碼完全不用改)
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(heatmap_data, cmap="viridis", aspect="auto")
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label("Average Attention Weight")
    
    ax.set_yticks(np.arange(len(modalities)))
    ax.set_yticklabels(modalities)
    ax.set_xticks(np.arange(K))
    ax.set_xticklabels(np.arange(1, K + 1)) # 月份 1 到 12
    
    ax.set_xlabel("Time Step (Month)")
    ax.set_ylabel("Modality")
    ax.set_title(title)
    
    # 在格子裡顯示數字
    for i in range(len(modalities)):
        for j in range(K):
            text = ax.text(j, i, f"{heatmap_data[i, j]:.2f}",
                           ha="center", va="center", color="w" if heatmap_data[i, j] < 0.5 else "k")
            
    fig.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

# Train/Eval

def move_inputs(batch: dict, device: torch.device, target: str):
    to = lambda t: (t.float().to(device) if isinstance(t, torch.Tensor) else t)
    price   = to(batch["price"]);   finance = to(batch["finance"])
    event   = to(batch["event"]);   news    = to(batch["news"])
    adj = to (batch["network"]); company_id = to(batch["company_id"])

    if price.ndim == 3: price = price.unsqueeze(0)
    if finance.ndim == 3: finance = finance.unsqueeze(0)
    if event.ndim == 3: event = event.unsqueeze(0)
    if news.ndim == 3: news = news.unsqueeze(0)
    if adj.ndim ==3: adj = adj.unsqueeze(0)

    bd = {"price":price, "finance":finance, "news":news, "event":event, "network": adj, "company_id": company_id}
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
    log = {"loss_total":0.0,"mse":0.0,"ic_company":0.0,"cl":0.0,"steps":0}
    

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
            log["cl"] += float(losses.get("cl", 0.0))
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
             desc="val",
             plot_attn=False) -> Tuple[Dict[str, float], Dict[int, dict]]:
    """
    評估使用「公司層級」：
      - 取 pred_company [B,1,N,1] 與 label_company [B,1,N,1]
      - 將 B 與 N 攤平成一維，對所有公司計算 mse/mae/rmse/smape
      - ic_company 也沿公司維度計算（pearson over all companies across years）
    per_year[y] 會存該年的「公司層級」向量（preds/labels 長度 ~ N * B）
    """
    model.eval()
    metrics = {"mse":0.0, "mae":0.0, "rmse":0.0, "smape":0.0, "ic_company": 0.0, "cl": 0.0, "count":0}
    per_year = {}

    all_preds_company, all_labels_company = [], []
    all_alphas = {"price": [], "fin": [], "news": [], "event": []}
    has_alphas = True
    years = sorted(list(loaders_by_year.keys()))
    for y in years:
        preds_y_list, labels_y_list = [], []
        symbols_y: List[str] = []

        for raw in loaders_by_year[y]:
            batch = move_inputs(raw, device, args.target)
            out = model(batch)

            if ("label_company" not in batch) or ("pred_company" not in out):
                break

            pc = out["pred_company"].squeeze(1).squeeze(-1).detach().cpu().numpy()  # [B,N]
            lc = batch["label_company"].squeeze(1).squeeze(-1).detach().cpu().numpy()  # [B,N]

            # 攤平成向量（B*N）
            preds_y_list.append(pc.reshape(-1))
            labels_y_list.append(lc.reshape(-1))

            if plot_attn:
                try:
                    all_alphas["price"].append(out["alpha_time_price"].cpu().numpy())
                    all_alphas["fin"].append(out["alpha_time_fin"].cpu().numpy())
                    all_alphas["news"].append(out["alpha_time_news"].cpu().numpy())
                    all_alphas["event"].append(out["alpha_time_event"].cpu().numpy())
                except KeyError as e:
                    if has_alphas: 
                        print(f"Warning: Model output missing key ({e}). No attention plot.")
                    has_alphas = False
                    plot_attn = False

            # raw取symbols
            syms = raw.get("symbols", None)
            if syms is not None and len(symbols_y) == 0:
                if isinstance(syms, list):
                    symbols_y = syms
                else:
                    try:
                        symbols_y = list(syms)
                    except Exception:
                        pass

        if len(preds_y_list) == 0:
            break

        preds_y = np.concatenate(preds_y_list)   # [~B*N]
        labels_y = np.concatenate(labels_y_list)  # [~B*N]

        se_y  = (labels_y - preds_y) ** 2
        sse_y = float(se_y.sum())    
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
        
        Yh = all_preds_company
        Y  = all_labels_company
        rmse_model = float(np.sqrt(np.mean((Y - Yh)**2)))
        rmse_mean  = float(np.sqrt(np.mean((Y - Y.mean())**2)))
        # print(f"[DBG2][{desc}] RMSE(model)={rmse_model:.4f} vs RMSE(mean)={rmse_mean:.4f}")
        # print(f"[DBG3][{desc}] pred std={float(Yh.std()):.4f}  true std={float(Y.std()):.4f}")

    if plot_attn and has_alphas and len(all_alphas["price"]) > 0:
        plot_path = os.path.join("/home/sally/myWork/SwitchAttention/outputs/heatmap", f"{desc}_temporal_attention_{args.target}.png")
        try:

            plot_temporal_attention_heatmap(
                all_alphas, 
                plot_path, 
                title=f"Average Temporal Attention ({desc})"
            )
            print(f"Saved attention heatmap to {plot_path}")
        except Exception as e:
            print(f"Failed to plot attention heatmap. Error: {e}")
            import traceback
            traceback.print_exc()

    return metrics, per_year


# Main
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
    # dimension
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
        "train": {"loss_total": [], "mse": [], "ic_company": [], "cl": [],},
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
        history["train"]["cl"].append(tr_log["cl"])

        # validation
        val_metrics, val_detail = evaluate(model, val_loaders, device, args, desc="val")
        for k in ["mse","mae","rmse","smape","ic_company"]:
            history["val"].setdefault(k, []).append(val_metrics.get(k, float("nan")))

        # 以 val MSE 當監控指標
        cur = val_metrics["mse"]
        improved = (best_val - cur) > args.early_stop_min_delta

        if improved:
            best_val = cur
            best_epoch = epoch
            no_improve = 0
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "val_mse": best_val}, best_path)
        else:
            no_improve += 1

        # 畫曲線 & 印 log
        plot_curves(history, os.path.join(args.out_dir, f"curves_{args.target}.png"))
        print(
            f"Epoch {epoch:03d} | Train total {tr_log['loss_total']:.4f} "
            f"(mse {tr_log['mse']:.4f}, ic_company {tr_log['ic_company']:.4f}), cl_loss {tr_log['cl']} | "
            f"Val MSE {val_metrics['mse']:.4f} RMSE {val_metrics['rmse']:.4f} "
            f"ic_company {val_metrics.get('ic_company', float('nan')):.4f} | "
            f"no_improve={no_improve}/{args.early_stop_patience}"
        )

        # Early Stopping
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
        #     f"(mse {tr_log['mse']:.4f}, ic_company {tr_log['ic_company']:.4f}), cl_loss {tr_log['cl']} | "
        #     f"Val MSE {val_metrics['mse']:.4f} RMSE {val_metrics['rmse']:.4f} "
        #     f"IC_company {val_metrics.get('ic_company', float('nan')):.4f}"
        # )
        
    print(f"[INFO] best_val after training = {best_val:.6f}")
    # Load best and plot once for VAL
    if os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded best model from epoch {ckpt['epoch']} with val_mse={ckpt['val_mse']:.6f}")

    val_metrics, val_detail = evaluate(model, val_loaders, device, args, desc="val(best)", plot_attn=True)    
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
    test_metrics, test_detail = evaluate(model, test_loaders, device, args, desc="test", plot_attn=True)   
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

    compute_and_save_baselines(
        train_loaders=train_loaders,
        val_loaders=val_loaders,
        test_loaders=test_loaders,
        out_dir="/home/sally/myWork/SwitchAttention",     
        split_names=("val","test"),
        target=args.target  
    )


if __name__ == "__main__":
    main()
