import os
import math
import time
import argparse
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn

# local modules (assume these exist)
from dataset_v2 import GraphESGDataset
from model import DynScan
from dataloader import build_loaders

# -----------------------
# default hyperparams (can be overridden by CLI)
# -----------------------
SEED = 2025
GRAD_CLIP = 1.0

# -----------------------
# utilities
# -----------------------
def set_seed(seed: int = SEED):
    import random, numpy as _np
    random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def ensure_batch_dim(t: torch.Tensor) -> torch.Tensor:
    """
    If tensor shape is [K,N,D] (3D), make it [B=1,K,N,D].
    If already has batch dim (4D), return as-is.
    """
    if t is None:
        return None
    if not torch.is_tensor(t):
        return t
    if t.dim() == 3:
        return t.unsqueeze(0)
    return t

def move_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out

def iterate_loader(loader):
    """
    Unified iterator:
    - If loader is dict: {year: DataLoader}, yields batches and sets batch['year']=year if missing.
    - If loader is DataLoader (or iterable), yields batches directly.
    """
    if isinstance(loader, dict):
        for year, dl in loader.items():
            for b in dl:
                if "year" not in b or b["year"] is None:
                    # ensure it's a tensor
                    b["year"] = torch.tensor(year)
                yield b
    else:
        for b in loader:
            # if loader yields (year, batch) tuple, handle that case
            if isinstance(b, tuple) and len(b) == 2 and isinstance(b[0], int):
                year, batch = b
                if "year" not in batch or batch["year"] is None:
                    batch["year"] = torch.tensor(year)
                yield batch
            else:
                yield b

# metrics
@torch.no_grad()
def smape(pred, target, eps=1e-8):
    pred = pred.detach().cpu()
    target = target.detach().cpu()
    return (100 * torch.mean(
        torch.abs(pred - target) / ((torch.abs(pred) + torch.abs(target)) / 2 + eps)
    )).item()

def compute_ic(preds_arr: np.ndarray, labels_arr: np.ndarray) -> float:
    if len(preds_arr) > 1:
        try:
            return float(np.corrcoef(preds_arr, labels_arr)[0,1])
        except Exception:
            return float("nan")
    return float("nan")

# -----------------------
# evaluation (single function for both overall & per-year)
# -----------------------
def evaluate(model: nn.Module, loader, device: torch.device, tidx: int, per_year: bool = False):
    model.eval()
    total_loss = mse_sum = mae_sum = smape_sum = 0.0
    n = 0
    preds_list = []
    labels_list = []

    # per-year buckets if requested
    buckets = defaultdict(lambda: {"mse": [], "mae": [], "smape": [], "preds": [], "labels": []})

    with torch.no_grad():
        for batch in iterate_loader(loader):
            # move and ensure batch dims
            batch = move_to_device(batch, device)
            # ensure tensors have batch dim [B,K,N,D] - model expects batch dim
            price   = ensure_batch_dim(batch["price"]).to(device)
            finance = ensure_batch_dim(batch["finance"]).to(device)
            event   = ensure_batch_dim(batch["event"]).to(device)
            network = ensure_batch_dim(batch["network"]).to(device)
            news    = ensure_batch_dim(batch.get("news", None))
            if news is not None:
                news = news.to(device)
            label = ensure_batch_dim(batch["label"]).to(device)
            label = label[..., tidx:tidx+1]  # [B,N,1] or [1,N,1]

            pred, slots, _ = model(price, finance, network, event, news)
            loss_total, loss_mse, loss_inter, loss_intra = model.compute_losses(pred, label, slots)

            mae = torch.abs(pred - label).mean().item()
            smape_val = smape(pred, label)

            total_loss += loss_total.item()
            mse_sum += loss_mse.item()
            mae_sum += mae
            smape_sum += smape_val
            inter_sum += loss_inter.item()
            intra_sum += loss_intra.item()
            n += 1

            # flatten to 1d arrays for IC
            preds_list.append(pred.cpu().numpy().ravel())
            labels_list.append(label.cpu().numpy().ravel())

            if per_year:
                year = int(batch.get("year", -1))
                buckets[year]["mse"].append(loss_mse.item())
                buckets[year]["mae"].append(mae)
                buckets[year]["smape"].append(smape_val)
                buckets[year]["preds"].append(pred.cpu().numpy().ravel())
                buckets[year]["labels"].append(label.cpu().numpy().ravel())

    # overall metrics
    preds = np.concatenate(preds_list) if preds_list else np.array([])
    labels = np.concatenate(labels_list) if labels_list else np.array([])
    ic = compute_ic(preds, labels)
    avg_mse = mse_sum / max(n, 1)
    avg_rmse = math.sqrt(avg_mse)
    avg_mae = mae_sum / max(n, 1)
    avg_smape = smape_sum / max(n, 1)
    avg_inter = inter_sum / max(n, 1)
    avg_intra = intra_sum / max(n, 1)
    avg_total = total_loss / max(n, 1)

    if not per_year:
        return avg_total, avg_mse, avg_rmse, avg_mae, avg_smape, avg_inter, avg_intra, ic
    else:
        # compute per-year summary
        results = {}
        for y, d in buckets.items():
            preds_y = np.concatenate(d["preds"]) if d["preds"] else np.array([])
            labels_y = np.concatenate(d["labels"]) if d["labels"] else np.array([])
            ic_y = compute_ic(preds_y, labels_y)
            mse_y = sum(d["mse"]) / max(len(d["mse"]), 1)
            mae_y = sum(d["mae"]) / max(len(d["mae"]), 1)
            smape_y = sum(d["smape"]) / max(len(d["smape"]), 1)
            rmse_y = math.sqrt(mse_y)
            results[y] = (mse_y, rmse_y, mae_y, smape_y, ic_y)
        return results

# -----------------------
# single epoch training
# -----------------------
def train_one_epoch(model, optimizer, scaler, train_loader, device, tidx, scheduler=None, epoch=0, grad_clip=GRAD_CLIP):
    model.train()
    running = {"total": 0.0, "mse": 0.0, "inter": 0.0, "intra": 0.0}
    count = 0
    pbar = tqdm(iterate_loader(train_loader), desc=f"Train Epoch {epoch:03d}")

    for batch in pbar:
        batch = move_to_device(batch, device)
        price   = ensure_batch_dim(batch["price"]).to(device)
        finance = ensure_batch_dim(batch["finance"]).to(device)
        event   = ensure_batch_dim(batch["event"]).to(device)
        network = ensure_batch_dim(batch["network"]).to(device)
        news    = ensure_batch_dim(batch.get("news", None))
        if news is not None:
            news = news.to(device)
        label = ensure_batch_dim(batch["label"]).to(device)
        label = label[..., tidx:tidx+1]

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=(device.type == "cuda")):
            pred, slots, _ = model(price, finance, network, event, news)
            loss_total, loss_mse, loss_inter, loss_intra = model.compute_losses(pred, label, slots)

        # after backward...
        scaler.scale(loss_total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step optimizer via scaler
        scaler.step(optimizer)
        scaler.update()

        running["total"] += loss_total.item()
        running["mse"] += loss_mse.item()
        running["inter"] += loss_inter.item()
        running["intra"] += loss_intra.item()
        count += 1

        pbar.set_postfix(total=f"{loss_total.item():.4f}", mse=f"{loss_mse.item():.4f}", inter=f"{loss_inter.item():.4f}", intra=f"{loss_intra.item():.4f}")

    # average
    for k in running:
        running[k] = running[k] / max(count, 1)
    return running

# -----------------------
# main
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["env", "soc", "gov"], required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--lambd", type=float, default=10.0)
    p.add_argument("--logdir", type=str, default="runs")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    tidx = {"env":0,"soc":1,"gov":2}[args.target]
    print(f"Using device: {device}, target: {args.target}")

    run_name = datetime.now().strftime("DynScan_%Y%m%d-%H%M%S")
    # logdir = os.path.join(args.logdir, f"dynscan_{args.target}_{run_name}")
    logdir = "/home/sally/myWork"
    os.makedirs(logdir, exist_ok=True)
    print(f"Logdir: {logdir}")

    # -------- dataset & loaders (uses your existing build_loaders) --------
    root_paths = {
        "price": "/home/sally/dataset/data_preprocessing/price_percentage",
        "finance": "/home/sally/dataset/data_preprocessing/financial",
        "news": "/home/sally/dataset/data_preprocessing/news/monthly_embeddings",
        "event": "/home/sally/dataset/data_preprocessing/event_type_PCA",
        "graph": "/home/sally/dataset/gkg_data/monthly_graph_new",
        "label": "/home/sally/dataset/data_preprocessing/esg_label/esg_npy",
        "year_symbols": "/home/sally/dataset/ticker/nyse/yearly_symbol"
    }

    train_loader, val_loader, test_loader = build_loaders(
        years_train=range(2015, 2020),
        years_val=range(2020, 2022),
        years_test=range(2022, 2025),
        batch_size=args.batch_size,
        root_paths=root_paths
    )

    # -------- model / optimizer / scheduler / amp --------
    # to infer dims, you can still instantiate dataset to get dims if needed
    sample_ds = GraphESGDataset(
        root_price=root_paths["price"],
        root_finance=root_paths["finance"],
        root_news=root_paths["news"],
        root_event=root_paths["event"],
        root_graph=root_paths["graph"],
        root_label=root_paths["label"],
        root_year_symbols=root_paths["year_symbols"],
        years=range(2015,2025),
        has_label=True,
        strict_check=True,
        fill_missing_event="zeros",
        fill_missing_graph="zeros",
    )
    model = DynScan(
        price_dim=sample_ds.price_dim,
        finance_dim=sample_ds.finance_dim,
        event_dim=sample_ds.event_dim,
        news_dim=sample_ds.news_dim,
        hidden_dim=32,
        num_slots=10,
        dropout=0.1,
        out_dim=1,
        alpha=args.alpha,
        lambd=args.lambd
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    # training history
    history = defaultdict(list)
    save_path = f"best_dynscan_{args.target}.pt"
    best_val = float("inf")
    best_epoch = -1

    # === DEBUG B: run one training step to inspect grads/param change ===
    def run_one_step_debug(model, optimizer, scaler, loader, device, tidx):
        print("=== DEBUG B: run_one_step_debug start ===")
        model.train()
        it = iter(iterate_loader(loader))
        batch = next(it)  # first batch
        batch = move_to_device(batch, device)
        price   = ensure_batch_dim(batch["price"]).to(device)
        finance = ensure_batch_dim(batch["finance"]).to(device)
        event   = ensure_batch_dim(batch["event"]).to(device)
        network = ensure_batch_dim(batch["network"]).to(device)
        news    = ensure_batch_dim(batch.get("news", None))
        if news is not None:
            news = news.to(device)
        label = ensure_batch_dim(batch["label"]).to(device)[..., tidx:tidx+1]

        # snapshot a small subset of params (first param tensor)
        first_param = next(model.parameters())
        w_before = first_param.detach().cpu().clone()

        # forward
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=(device.type == "cuda")):
            pred, slots, _ = model(price, finance, network, event, news)
            loss_total, loss_mse, loss_inter, loss_intra = model.compute_losses(pred, label, slots)

        print(f"[DEBUG B] loss_total={loss_total.item():.6f}, loss_mse={loss_mse.item():.6f}")
        # backward + step (use scaler same as training)
        scaler.scale(loss_total).backward()
        scaler.unscale_(optimizer)

        # grad norm
        total_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += float(p.grad.detach().norm().item())
        print(f"[DEBUG B] total_grad_norm (before clip) = {total_grad_norm:.6e}")

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        # snapshot again
        w_after = first_param.detach().cpu().clone()
        change_norm = float((w_after - w_before).norm().item())
        print(f"[DEBUG B] first_param change norm after step = {change_norm:.6e}")

        # show a few pred vs label examples
        pred_np = pred.detach().cpu().numpy().ravel()
        label_np = label.detach().cpu().numpy().ravel()
        for i in range(min(5, len(pred_np))):
            print(f"[DEBUG B] sample {i}: pred={pred_np[i]:.6f}, label={label_np[i]:.6f}, err={pred_np[i]-label_np[i]:.6f}")

        print("=== DEBUG B: run_one_step_debug end ===\n")
        # we intentionally do not modify loader state; training can continue

    # call it once before full training loop
    try:
        run_one_step_debug(model, optimizer, scaler, train_loader, device, tidx)
    except StopIteration:
        print("[DEBUG B] train_loader empty or iterate_loader yielded no batch.")
    except Exception as e:
        print("[DEBUG B] Exception during one-step debug:", e)


    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, optimizer, scaler, train_loader, device, tidx,
                                      scheduler=scheduler, epoch=epoch, grad_clip=GRAD_CLIP)

        val_total, val_mse, val_rmse, val_mae, val_smape, val_inter, val_intra, val_ic = evaluate(model, val_loader, device, tidx, per_year=False)
        train_total, train_mse, train_rmse, train_mae, train_smape, train_inter, train_intra, train_ic = evaluate(model, train_loader, device, tidx, per_year=False)

        # record
        history["epoch"].append(epoch)
        history["train_total"].append(train_total); history["train_mse"].append(train_mse); history["train_ic"].append(train_ic)
        history["val_total"].append(val_total); history["val_mse"].append(val_mse); history["val_ic"].append(val_ic)

        print(f"[Epoch {epoch}] train_total={train_total:.4f} val_total={val_total:.4f} val_ic={val_ic:.4f}")

        # checkpoint
        if val_total < best_val - 1e-8:
            best_val = val_total
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_total": val_total
            }, save_path)
            print(f">>> Saved best model @ epoch {epoch} (val_total={val_total:.4f})")

        # optionally free cuda mem
        torch.cuda.empty_cache()
        # train_one_epoch: remove any scheduler.step() calls inside here

        # step scheduler once per epoch (after optimizer has been stepping inside train_one_epoch)
        if scheduler is not None:
            scheduler.step()


    # save history
    df_hist = pd.DataFrame(history)
    hist_path = os.path.join(logdir, f"history_{args.target}.csv")
    df_hist.to_csv(hist_path, index=False)
    print(f"Saved history to {hist_path}")

    # load best and test + per-year
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded best ckpt epoch={ckpt['epoch']}, val_total={ckpt['val_total']:.4f}")

    test_metrics = evaluate(model, test_loader, device, tidx, per_year=False)
    print(f"[TEST] {test_metrics}")

    yearly_results = evaluate(model, test_loader, device, tidx, per_year=True)

    # save yearly results
    rows = []
    for y,(mse,rmse,mae,smape_val,ic) in sorted(yearly_results.items()):
        rows.append({"year": y, "mse": mse, "rmse": rmse, "mae": mae, "smape": smape_val, "ic": ic})
        print(f"Year {y}: mse={mse:.4f} rmse={rmse:.4f} mae={mae:.4f} smape={smape_val:.4f} ic={ic:.4f}")
    pd.DataFrame(rows).to_csv(os.path.join(logdir, f"yearly_{args.target}.csv"), index=False)

    # 儲存 Test predictions + labels -> CSV（每列: year, symbol, pred, label）
    # 放在載入 best checkpoint並跑完 test/evaluate 之後執行

    def _ensure_symbol_list(batch, N):
        """
        取得 symbols list（長度 N）。
        支援 batch["symbols"] 為 list / numpy array / torch tensor / None
        """
        cand_names = None
        for key in ["symbols", "symbol", "ticker", "tickers", "names"]:
            if key in batch and batch[key] is not None:
                cand_names = batch[key]
                break

        if cand_names is None:
            # fallback: use index-based names
            return [f"node_{i}" for i in range(N)]

        # convert different types to python list of str
        if isinstance(cand_names, list):
            names = list(cand_names)
        elif isinstance(cand_names, np.ndarray):
            names = cand_names.tolist()
        elif torch.is_tensor(cand_names):
            try:
                # if tensor of strings (rare), convert to numpy
                names = cand_names.cpu().numpy().tolist()
            except Exception:
                # numeric tensor: treat as indices
                names = [str(int(x)) for x in cand_names.cpu().numpy().ravel()]
        else:
            # unknown type: coerce to list
            try:
                names = list(cand_names)
            except Exception:
                names = [str(cand_names)] * N

        # sometimes symbols are per-node w/o batch dim or per-batch; ensure length N
        # If names is nested (e.g., per-batch list), try to unwrap
        if len(names) == 0:
            return [f"node_{i}" for i in range(N)]
        if len(names) == N:
            return [str(x) for x in names]
        # if shapes like [B, N] where B=1
        if len(names) == 1 and isinstance(names[0], (list, np.ndarray)):
            sub = names[0]
            if len(sub) == N:
                return [str(x) for x in sub]
        # otherwise fallback to index labels
        return [f"node_{i}" for i in range(N)]

    # 實作：collect all preds/labels/symbol/year -> DataFrame -> CSV
    def save_test_predictions_csv(model, test_loader, device, tidx, save_path="preds_labels.csv"):
        model.eval()
        rows = []
        with torch.no_grad():
            # 支援 test_loader 為 dict(year -> DataLoader) 或單一 DataLoader
            if isinstance(test_loader, dict):
                for year, dl in test_loader.items():
                    for batch in dl:
                        # move to device & ensure batch dim
                        batch_dev = move_to_device(batch, device)
                        price   = ensure_batch_dim(batch_dev["price"]).to(device)
                        finance = ensure_batch_dim(batch_dev["finance"]).to(device)
                        event   = ensure_batch_dim(batch_dev["event"]).to(device)
                        network = ensure_batch_dim(batch_dev["network"]).to(device)
                        news    = ensure_batch_dim(batch_dev.get("news", None))
                        if news is not None:
                            news = news.to(device)
                        label = ensure_batch_dim(batch_dev["label"]).to(device)[..., tidx:tidx+1]  # [B,N,1]

                        pred, slots, _ = model(price, finance, network, event, news)  # pred: [B,N,1]

                        # flatten B dimension (we usually have B=1)
                        pred_np = pred.cpu().numpy().reshape(-1)   # (B*N,)
                        label_np = label.cpu().numpy().reshape(-1) # (B*N,)

                        # attempt to get symbols (per-node)
                        # original batch may have symbols without batch dim (N,) or with batch dim (B,N)
                        # we prefer `batch["symbols"]` from original batch (not moved to device)
                        symbols = _ensure_symbol_list(batch, N=pred_np.shape[0] if pred_np.shape[0] > 0 else pred.shape[1])

                        # If symbols length != number of nodes, try to derive from shape (B,N) case:
                        if len(symbols) != pred_np.shape[0]:
                            # if original batch has batch dim and B==1, unwrap
                            if "symbols" in batch and hasattr(batch["symbols"], "__len__"):
                                try:
                                    # try to flatten and force length
                                    flat = np.array(batch["symbols"]).ravel().tolist()
                                    if len(flat) >= pred_np.shape[0]:
                                        symbols = [str(x) for x in flat[:pred_np.shape[0]]]
                                except Exception:
                                    pass

                        # ensure length again
                        if len(symbols) != pred_np.shape[0]:
                            # fallback to node_{i}
                            symbols = [f"node_{i}" for i in range(pred_np.shape[0])]

                        # append rows
                        for i, sym in enumerate(symbols):
                            rows.append({
                                "year": int(year),
                                "symbol": str(sym),
                                "pred": float(pred_np[i]),
                                "label": float(label_np[i]),
                                "error": float(pred_np[i]-label_np[i])
                            })
            else:
                # loader not dict: iterate and try to use batch["year"] if present
                for batch in test_loader:
                    batch_dev = move_to_device(batch, device)
                    price   = ensure_batch_dim(batch_dev["price"]).to(device)
                    finance = ensure_batch_dim(batch_dev["finance"]).to(device)
                    event   = ensure_batch_dim(batch_dev["event"]).to(device)
                    network = ensure_batch_dim(batch_dev["network"]).to(device)
                    news    = ensure_batch_dim(batch_dev.get("news", None))
                    if news is not None:
                        news = news.to(device)
                    label = ensure_batch_dim(batch_dev["label"]).to(device)[..., tidx:tidx+1]

                    pred, slots, _ = model(price, finance, network, event, news)

                    pred_np = pred.cpu().numpy().reshape(-1)
                    label_np = label.cpu().numpy().reshape(-1)

                    year = int(batch.get("year", -1)) if "year" in batch else -1
                    symbols = _ensure_symbol_list(batch, N=pred_np.shape[0])

                    if len(symbols) != pred_np.shape[0]:
                        symbols = [f"node_{i}" for i in range(pred_np.shape[0])]

                    for i, sym in enumerate(symbols):
                        rows.append({
                            "year": year,
                            "symbol": str(sym),
                            "pred": float(pred_np[i]),
                            "label": float(label_np[i]),
                            "errors": float(pred_np[i]-label[i])
                        })

        # save DataFrame
        if len(rows) == 0:
            print("[save_test_predictions_csv] Warning: no rows collected; CSV not written.")
            return

        df = pd.DataFrame(rows)
        # 排序：先 year 再 symbol
        df = df.sort_values(["year", "symbol"]).reset_index(drop=True)
        df.to_csv(save_path, index=False)
        df["abs_error"] = (df["pred"]-df["label"]).abs()
        print(f"[save_test_predictions_csv] saved {len(df)} rows to {save_path}")

    save_csv_path = f"preds_labels_{args.target}.csv"
    save_test_predictions_csv(model, test_loader, device, tidx, save_path=save_csv_path)


    # scatter plot (outputs vs labels)
    all_out, all_label = [], []
    model.eval()
    with torch.no_grad():
        for batch in iterate_loader(test_loader):
            batch = move_to_device(batch, device)
            price = ensure_batch_dim(batch["price"]).to(device)
            finance = ensure_batch_dim(batch["finance"]).to(device)
            event = ensure_batch_dim(batch["event"]).to(device)
            network = ensure_batch_dim(batch["network"]).to(device)
            news = ensure_batch_dim(batch.get("news", None))
            if news is not None: news = news.to(device)
            label = ensure_batch_dim(batch["label"]).to(device)[..., tidx:tidx+1]
            pred, *_ = model(price, finance, network, event, news)
            all_out.append(pred.cpu().view(-1))
            all_label.append(label.cpu().view(-1))
    if all_out:
        all_out = torch.cat(all_out).numpy()
        all_label = torch.cat(all_label).numpy()
        plt.figure(figsize=(6,6))
        plt.scatter(all_label, all_out, alpha=0.5)
        mn = min(all_label.min(), all_out.min())
        mx = max(all_label.max(), all_out.max())
        plt.plot([mn,mx],[mn,mx],'r--')
        plt.xlabel("True"); plt.ylabel("Pred")
        plt.title(f"Scatter {args.target}")
        plt.savefig(os.path.join(logdir, f"scatter_{args.target}.png"), dpi=300)
        plt.close()
        print(f"Saved scatter to {os.path.join(logdir, f'scatter_{args.target}.png')}")
    print("model first param device:", next(model.parameters()).device)

    

if __name__ == "__main__":
    main()
