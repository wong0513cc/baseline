# dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

ROOT_PRICE="/home/sally/dataset/data_preprocessing/price_percentage"
ROOT_FIN="/home/sally/dataset/data_preprocessing/financial"
ROOT_NEWS="/home/sally/dataset/data_preprocessing/news/monthly_embeddings"
ROOT_EVENT="/home/sally/dataset/data_preprocessing/event_type_PCA"
ROOT_GRAPH="/home/sally/dataset/gkg_data/monthly_graph_new"
ROOT_LABEL="/home/sally/dataset/data_preprocessing/esg_label/esg_npy"
ROOT_SYMS="/home/sally/dataset/ticker/nyse/yearly_symbol"

class GraphESGDataset(Dataset):
    """
    每個樣本＝一個年份
    回傳 keys:
      - price:   (12, N, d_p)
      - finance: (12, N, d_f)
      - news:    (12, N, d_n)
      - event:   (12, N, d_e)   # 由 12 個月檔案堆疊
      - network: (12, N, N)     # 由 12 個月 adjacency 檔案堆疊
      - label:   (N, 3)         # optional
      - symbols: (N,)           # 當年公司順序（除錯/記錄用）
    """
    def __init__(
        self,
        root_price,            #/data/price
        root_finance,          #/data/finance
        root_news,             #/data/news
        root_event,            #/data/event_type
        root_graph,            #/data/monthly_graph
        root_label=None,       #/data/labels
        root_year_symbols=None,#  /data/year_symbol_list_csv
        years=range(2015, 2025),
        has_label=True,
        event_dim=64,
        strict_check=True,     # 檢查所有模態的 N 是否一致
        fill_missing_event="zeros",  # or "nan"
        fill_missing_graph="zeros"   # or "nan"
    ):
        self.root_price = root_price
        self.root_finance = root_finance
        self.root_news = root_news
        self.root_event = root_event
        self.root_graph = root_graph
        self.root_label = root_label
        self.root_year_symbols = root_year_symbols
        self.years = list(years)
        self.has_label = has_label
        self.event_dim = event_dim
        self.strict_check = strict_check
        self.fill_missing_event = fill_missing_event
        self.fill_missing_graph = fill_missing_graph

        self.samples = []
        for y in self.years:
            sample = self._load_one_year(y)
            self.samples.append(sample)

            # 初始化特徵維度
        first = self.samples[0]
        self.price_dim   = first["price"].shape[2]   # d_p
        self.finance_dim = first["finance"].shape[2] # d_f
        self.event_dim   = first["event"].shape[2]   # d_e
        self.news_dim    = first["news"].shape[2]


    def _load_year_symbols(self, year):
        if self.root_year_symbols is None:
            return None
        path = os.path.join(self.root_year_symbols, f"{year}_symbol.csv")
        syms = pd.read_csv(path)["symbol"].astype(str).tolist()
        return syms

    def _stack_event(self, year, N):
        months = [f"{m:02d}" for m in range(1, 13)]
        seq = []
        for mm in months:
            fpath = os.path.join(self.root_event, f"event_{year}-{mm}_pca64.npy")
            if os.path.exists(fpath):
                arr = np.load(fpath)  # (N?, d_e)
                # 若 N 不同，直接拋錯比較安全
                if arr.shape[1] != self.event_dim:
                    raise ValueError(f"[event] dim mismatch at {fpath}, got {arr.shape}")
                if arr.shape[0] != N:
                    raise ValueError(f"[event] N mismatch at {fpath}: got {arr.shape[0]} vs expected {N}")
                seq.append(arr)
            else:
                if self.fill_missing_event == "zeros":
                    seq.append(np.zeros((N, self.event_dim), dtype=np.float32))
                else:
                    seq.append(np.full((N, self.event_dim), np.nan, dtype=np.float32))
        return np.stack(seq, axis=0)  # (12, N, d_e)

    def _stack_graph(self, year, N):
        months = [f"{m:02d}" for m in range(1, 13)]
        seq = []
        for mm in months:
            fpath = os.path.join(self.root_graph, f"adj_{year}", f"adj_{year}-{mm}.npy")
            if os.path.exists(fpath):
                A = np.load(fpath)  # (N?, N?)
                if A.shape[0] != N or A.shape[1] != N:
                    raise ValueError(f"[graph] N mismatch at {fpath}: got {A.shape} vs expected {(N, N)}")
                seq.append(A.astype(np.float32))
            else:
                if self.fill_missing_graph == "zeros":
                    seq.append(np.zeros((N, N), dtype=np.float32))
                else:
                    seq.append(np.full((N, N), np.nan, dtype=np.float32))
        return (np.stack(seq, axis=0) > 0).astype(np.bool_) 

    def _load_one_year(self, year):
        # 讀當年的 symbol list（用來確認 N 與除錯）
        symbols = self._load_year_symbols(year)

        # price / finance / news 皆已是 (12, N, dim)
        price = np.load(os.path.join(self.root_price,   f"price_pct_{year}.npy"))      # (12, N, d_p)
        news =  np.load(os.path.join(self.root_news,    f"news_{year}.npy"))
        # Finance: load dict
        finance_path = os.path.join(self.root_finance, f"financial_{year}.npy")
        finance_dict = np.load(finance_path, allow_pickle=True).item()

        finance = finance_dict["finance"]  # (12, N, d_f)
        finance_mask = finance_dict["mask"]
        finance_symbols = list(finance_dict["symbols"])

        if symbols is not None and finance_symbols != symbols:
            print(f"[警告] {year} finance symbols 與 year_symbol_list 不一致")

        N_price = price.shape[1]
        N_fin   = finance.shape[1]
        N_news  = news.shape[1]

        if self.strict_check and not (N_price == N_fin == N_news):
            raise ValueError(f"[{year}] N mismatch among (price={N_price}, finance={N_fin}, news={N_news})")

        N = N_price
        if symbols is not None and len(symbols) != N:
            raise ValueError(f"[{year}] symbols length {len(symbols)} != N {N}")

        # event (12, N, d_e) 由月檔堆疊
        event = self._stack_event(year, N)

        # graph (12, N, N) 由月檔堆疊
        network = self._stack_graph(year, N)

        sample = {
            "price":   torch.tensor(price, dtype=torch.float32),
            "finance": torch.tensor(finance, dtype=torch.float32),
            "finance_mask": torch.tensor(finance_mask, dtype=torch.float32),
            "news":    torch.tensor(news, dtype=torch.float32),
            "event":   torch.tensor(event, dtype=torch.float32),
            "network": torch.tensor(network, dtype=torch.bool),
            "symbols": symbols if symbols is not None else None,
            "year": year
        }

    

        if self.has_label and self.root_label is not None:
            label_path = os.path.join(self.root_label, f"{year}_esg.npy")
            if os.path.exists(label_path):
                label = np.load(label_path)  # (N, 3) with possible NaN
                if label.shape[0] != N:
                    raise ValueError(f"[{year}] label N {label.shape[0]} != {N}")
                
                label = label /100.0
                sample["label"] = torch.tensor(label, dtype=torch.float32) 
            else:
                sample["label"] = None

        # 檢查所有 N 是否一致
        if self.strict_check:
            N_event   = event.shape[1]
            N_graph   = network.shape[1]
            if not (N == N_event == N_graph):
                raise ValueError(f"[{year}] N mismatch after stacking: N={N}, eventN={N_event}, graphN={N_graph}")

        return sample

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample["year"] = torch.tensor(self.years[idx], dtype=torch.int32)
        return sample
years = range(2015,2025)

# ds = GraphESGDataset(
#                 root_price=ROOT_PRICE,
#                 root_finance=ROOT_FIN,
#                 root_news=ROOT_NEWS,
#                 root_event=ROOT_EVENT,
#                 root_graph=ROOT_GRAPH,
#                 root_label=ROOT_LABEL,
#                 root_year_symbols=ROOT_SYMS,
#                 years = range(2015,2025),
#                 has_label=True,
#                 strict_check=True,
#                 fill_missing_event="zeros",
#                 fill_missing_graph="zeros",
#             )
# sample = ds[1]
# label = sample.get("label", None)
# print(label.shape)