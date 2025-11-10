# industry_mapper.py
import os
import pandas as pd

class IndustryMapper:
    def __init__(self, root_year_symbols):
        """
        root_year_symbols 內含每年檔：
          - {year}_symbol_id_with_industry.csv   或
          - {year}_symbol_with_industry.csv
        欄位至少要有 symbol(or ticker), industry_id
        """
        self.root = root_year_symbols
        self.maps = {}   # year -> dict(symbol->industry_id)

    def _load_one_year(self, year: int):
        if year in self.maps:
            return self.maps[year]

        # 嘗試兩種常見檔名
        cands = [
            os.path.join(self.root, f"{year}_symbol_id_with_industry.csv"),
            os.path.join(self.root, f"{year}_symbol_with_industry.csv"),
        ]
        path = next((p for p in cands if os.path.exists(p)), None)
        if path is None:
            raise FileNotFoundError(f"找不到 {year} 的 yearly symbol 檔（帶 industry_id）")

        df = pd.read_csv(path, dtype=str)
        key = "symbol" if "symbol" in df.columns else ("ticker" if "ticker" in df.columns else None)
        if key is None:
            raise ValueError(f"{path} 缺少 symbol/ticker 欄位")
        df["sym"] = df[key].astype(str).str.strip().str.upper()

        if "industry_id" not in df.columns:
            raise ValueError(f"{path} 缺少 industry_id 欄位")

        mp = dict(
            zip(df["sym"], pd.to_numeric(df["industry_id"], errors="coerce").fillna(-1).astype(int))
        )
        self.maps[year] = mp
        return mp

    def get_ids(self, year: int, symbols):
        """
        symbols: list[str]（當年公司順序）
        回傳：list[int]（對不到給 -1）
        """
        mp = self._load_one_year(int(year))
        return [mp.get(str(s).strip().upper(), -1) for s in symbols]
