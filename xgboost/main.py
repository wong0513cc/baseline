import os
import numpy as np
import pandas as pd

from dataset import FinancialESGXGBDataset
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return np.mean(2.0 * np.abs(y_pred - y_true) / denom)


def pearson_ic(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    y_true = y_true - y_true.mean()
    y_pred = y_pred - y_pred.mean()

    num = np.sum(y_true * y_pred)
    denom = (np.sqrt(np.sum(y_true ** 2)) * np.sqrt(np.sum(y_pred ** 2)) + eps)

    return float(num / denom)


def main():
    years = list(range(2015, 2025))

    dataset = FinancialESGXGBDataset(
        financial_dir="/home/sally/baseline/dataset/financial_indicator",
        esg_label_csv="/home/sally/dataset/data_preprocessing/esg_label/NYSE_esg_score_final_v3.csv",
        years=years,
        target="Governance", 
    )


    train_years = [2015, 2016, 2017, 2018, 2019]
    val_years   = [2020, 2021]
    test_years  = [2023, 2024] 


    (train_X, train_y), (val_X, val_y), (test_X, test_y) = dataset.train_val_test_split_by_year(
        train_years=train_years,
        val_years=val_years,
        test_years=test_years,
    )
    train_y = train_y / 100.0
    val_y   = val_y / 100.0
    test_y  = test_y / 100.0

    print("Train shape:", train_X.shape, train_y.shape)
    print("Val shape:",   val_X.shape,   val_y.shape)
    print("Test shape:",  test_X.shape,  test_y.shape)

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=1e-3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=8,
    )

    model.fit(
        train_X, train_y,
        eval_set=[(val_X, val_y)],
        verbose=True,
    )

    # evaluate
    df = dataset.table.copy()
    feat_cols = dataset.feature_cols
    target_col = dataset.target   # "Environmental"

    results = []

    for year in test_years:
        mask = df["year"] == year
        X_year = df.loc[mask, feat_cols].to_numpy(dtype=np.float32)
        y_year = df.loc[mask, target_col].to_numpy(dtype=np.float32) / 100.0  # label /100

        y_pred = model.predict(X_year)

        mse = mean_squared_error(y_year, y_pred)
        mae = mean_absolute_error(y_year, y_pred)
        rmse = float(np.sqrt(mse))
        smape_val = smape(y_year, y_pred)
        ic_val = pearson_ic(y_year, y_pred)

        print(f"Year {year} | MSE={mse:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}, SMAPE={smape_val:.6f}, IC={ic_val:.6f}")

        results.append({
            "year": year,
            "target": target_col,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "smape": smape_val,
            "ic": ic_val,
            "num_samples": X_year.shape[0],
        })

    results_df = pd.DataFrame(results)

    out_dir = "/home/sally/baseline/results"
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"xgb_{target_col}_test_metrics.csv")
    results_df.to_csv(out_path, index=False)
    print(f"Saved metrics to: {out_path}")


if __name__ == "__main__":
    main()
