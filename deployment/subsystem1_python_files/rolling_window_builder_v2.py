import os
import pandas as pd
import numpy as np


def build_rolling_classification_dataset(
    df: pd.DataFrame,
    window: int = 28,
    step: int = 7,
    horizon_window: int = 28,
    long_window: int = 90,
    increase_threshold: float = 0.05,
    eps: float = 1e-6
):
    data = df.copy()
    data["OrderDate"] = pd.to_datetime(data["OrderDate"], errors="coerce")
    data = data.dropna(subset=["OrderDate"])

    data = data[data["Status"].astype(str).str.lower() == "shipped"].copy()
    data["OrderItemQuantity"] = pd.to_numeric(data["OrderItemQuantity"], errors="coerce").fillna(0)

    if data.empty:
        raise ValueError("No shipped records after cleaning.")

    daily = (
        data.groupby(["OrderDate", "CategoryName"], as_index=False)["OrderItemQuantity"]
        .sum()
        .rename(columns={"OrderItemQuantity": "demand"})
    )

    min_date = daily["OrderDate"].min()
    max_date = daily["OrderDate"].max()
    full_dates = pd.date_range(min_date, max_date, freq="D")

    X_rows, y_rows, meta_rows = [], [], []
    categories = daily["CategoryName"].unique()

    for cat in categories:
        s = daily[daily["CategoryName"] == cat].set_index("OrderDate")[["demand"]]
        s = s.reindex(full_dates, fill_value=0).reset_index().rename(columns={"index": "OrderDate"})
        s["CategoryName"] = cat

        total_needed = window + horizon_window
        if len(s) < total_needed:
            continue

        for start_i in range(0, len(s) - total_needed + 1, step):
            cur = s.iloc[start_i:start_i + window]["demand"]
            fut = s.iloc[start_i + window:start_i + window + horizon_window]["demand"]

            cur_mean = cur.mean()
            fut_mean = fut.mean()

            y = 1 if fut_mean > cur_mean * (1 + increase_threshold) else 0

            cur_sum = cur.sum()
            cur_std = cur.std(ddof=0)
            coverage = (cur > 0).mean()
            volatility_ratio = cur_std / (cur_mean + eps)

            half = window // 2
            first_half_mean = cur.iloc[:half].mean()
            second_half_mean = cur.iloc[half:].mean()
            intra_growth = (second_half_mean - first_half_mean) / (first_half_mean + eps)

            long_start_i = max(0, start_i + window - long_window)
            long_hist = s.iloc[long_start_i:start_i + window]["demand"]
            long_mean = long_hist.mean()
            long_sum = long_hist.sum()
            long_coverage = (long_hist > 0).mean()

            recent_nonzero = np.where(cur.to_numpy() > 0)[0]
            if len(recent_nonzero) == 0:
                days_since_last_sale = float(window)
            else:
                days_since_last_sale = float(window - 1 - recent_nonzero[-1])

            active_recent_flag = int(cur_sum > 0)
            fallback_baseline = max(float(cur_mean * window), float(long_mean * window))

            X_rows.append({
                "CategoryName": cat,
                "cur_mean": float(cur_mean),
                "cur_sum": float(cur_sum),
                "cur_std": float(cur_std),
                "coverage": float(coverage),
                "volatility_ratio": float(volatility_ratio),
                "intra_growth": float(intra_growth),
                "long_mean": float(long_mean),
                "long_sum": float(long_sum),
                "long_coverage": float(long_coverage),
                "days_since_last_sale": float(days_since_last_sale),
                "active_recent_flag": int(active_recent_flag),
                "fallback_baseline": float(fallback_baseline),
            })

            y_rows.append(y)

            meta_rows.append({
                "CategoryName": cat,
                "window_start": s.iloc[start_i]["OrderDate"],
                "window_end": s.iloc[start_i + window - 1]["OrderDate"],
                "future_start": s.iloc[start_i + window]["OrderDate"],
                "future_end": s.iloc[start_i + total_needed - 1]["OrderDate"]
            })

    X = pd.DataFrame(X_rows)
    y = pd.Series(y_rows, name="target_up")
    meta = pd.DataFrame(meta_rows)

    if not X.empty:
        X["intra_growth"] = X["intra_growth"].clip(-5, 5)
        X["volatility_ratio"] = X["volatility_ratio"].clip(0, 50)
        X["days_since_last_sale"] = X["days_since_last_sale"].clip(0, window)

    if not meta.empty:
        window_end = pd.to_datetime(meta["window_end"], errors="coerce")
        X["month"] = window_end.dt.month.astype("Int64")
        X["week_of_year"] = window_end.dt.isocalendar().week.astype("Int64")
        X["day_of_week"] = window_end.dt.dayofweek.astype("Int64")

    return X, y, meta


def build_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    X, y, meta = build_rolling_classification_dataset(df)
    return pd.concat([meta, X, y], axis=1)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    updated_base_path = os.path.join(base_dir, "data", "updated_base_history.csv")
    raw_path = os.path.join(base_dir, "data", "ML-Dataset.csv")

    if os.path.exists(updated_base_path):
        data = pd.read_csv(updated_base_path)
        print("Using updated base history")
    else:
        data = pd.read_csv(raw_path)
        print("Using original ML-Dataset")

    output_dir = os.path.join(base_dir, "data", "output")
    os.makedirs(output_dir, exist_ok=True)

    dataset = build_rolling_features(data)
    print("Generated dataset shape:", dataset.shape)
    print("Columns:", dataset.columns.tolist())
    dataset.to_csv(os.path.join(output_dir, "rolling_supervised_dataset_v2.csv"), index=False)

    print("Rolling window dataset generated.")