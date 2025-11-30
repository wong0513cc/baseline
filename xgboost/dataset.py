# xgb_dataset.py
import os
from typing import List, Tuple
import numpy as np
import pandas as pd


class FinancialESGXGBDataset:
    def __init__(
        self,
        financial_dir: str,
        esg_label_csv: str,
        years: List[int],
        target: str = "Environmental",  
    ):

        self.financial_dir = financial_dir
        self.years = list(years)
        self.target = target


        self.esg_df = pd.read_csv(esg_label_csv)

        assert "year" in self.esg_df.columns
        assert "symbol" in self.esg_df.columns
        assert target in self.esg_df.columns

        # year 轉成 int 比較保險
        self.esg_df["year"] = self.esg_df["year"].astype(int)
        self.esg_df["symbol"] = self.esg_df["symbol"].astype(str)

        # 建完整 table
        self.table = self._build_table()

        # feature 欄位名稱：feat_0, feat_1, ...
        self.feature_cols = [c for c in self.table.columns if c.startswith("feat_")]

    def _build_table(self) -> pd.DataFrame:

        all_rows = []

        for year in self.years:
            npy_path = os.path.join(self.financial_dir, f"financial_{year}.npy")

            data = np.load(npy_path, allow_pickle=True).item()
            finance = data["finance"]   # [N, F]
            symbols = data["symbols"]   # list len N

            F = finance.shape[1]
            feature_names = [f"feat_{j}" for j in range(F)]

            # [N,F] -> DataFrame
            df_fin = pd.DataFrame(finance, index=symbols, columns=feature_names)
            df_fin.index.name = "symbol"
            df_fin.reset_index(inplace=True)      # symbol 變成一欄
            df_fin["symbol"] = df_fin["symbol"].astype(str)
            df_fin["year"] = int(year)
            
            # esg
            df_esg_year = self.esg_df[self.esg_df["year"] == year][
                ["year", "symbol", self.target]
            ].copy()

            # merge on (year, symbol)
            df_merge = pd.merge(
                df_fin,
                df_esg_year,
                on=["year", "symbol"],
                how="inner",   
            )

            # df_merge.to_csv(f"/home/sally/baseline/merged_{year}.csv", index=False)

            all_rows.append(df_merge)

        if not all_rows:
            raise RuntimeError("沒有任何年份")

        table = pd.concat(all_rows, ignore_index=True)


        table = table.dropna(subset=[self.target])

        return table



    def get_xy(self) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        回傳:
          X: [num_samples, num_features]
          y: [num_samples]
          meta_df: 包含 year, symbol (方便之後分析 / 存結果)
        """
        meta_cols = ["year", "symbol"]
        meta_df = self.table[meta_cols].copy()

        X = self.table[self.feature_cols].to_numpy(dtype=np.float32)
        y = self.table[self.target].to_numpy(dtype=np.float32)

        return X, y, meta_df

    def train_val_test_split_by_year(
        self,
        train_years: List[int],
        val_years: List[int],
        test_years: List[int],
    ):

        df = self.table

        def _select(year_list):
            mask = df["year"].isin(year_list)
            X = df.loc[mask, self.feature_cols].to_numpy(dtype=np.float32)
            y = df.loc[mask, self.target].to_numpy(dtype=np.float32)
            return X, y

        X_train, y_train = _select(train_years)
        X_val,   y_val   = _select(val_years)
        X_test,  y_test  = _select(test_years)

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
