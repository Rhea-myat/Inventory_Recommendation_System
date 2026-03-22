import os
import pandas as pd
import numpy as np

def build_rolling_classification_dataset(
    df: pd.DataFrame,
    window: int = 28,
    step: int = 7,                 # step=7 gives weekly sliding windows 
    horizon_window: int = 28,      # future window length for label
    increase_threshold: float = 0.05,  # 5% threshold for "increase" label (balanced)
    eps: float = 1e-6
):
    """
    Build supervised dataset for demand-direction classification (per Category).
    
    Target (binary):
      y=1 if mean_future > mean_current * (1 + increase_threshold)
      y=0 otherwise
    
    Returns:
      X: DataFrame of features
      y: Series target labels
      meta: DataFrame with CategoryName, window_start, window_end (for time-based split)
    """

    data = df.copy()
    data["OrderDate"] = pd.to_datetime(data["OrderDate"], format="%d-%b-%y", errors="coerce")
    data = data.dropna(subset=["OrderDate"])

    # Shipped only
    data = data[data["Status"].astype(str).str.lower() == "shipped"].copy()

    # numeric qty
    data["OrderItemQuantity"] = pd.to_numeric(data["OrderItemQuantity"], errors="coerce").fillna(0)

    if data.empty:
        raise ValueError("No shipped records after cleaning.")

    # daily category demand
    daily = (
        data.groupby(["OrderDate", "CategoryName"], as_index=False)["OrderItemQuantity"]
            .sum()
            .rename(columns={"OrderItemQuantity": "demand"})
    )

    min_date = daily["OrderDate"].min()
    max_date = daily["OrderDate"].max()
    full_dates = pd.date_range(min_date, max_date, freq="D")

    X_rows = []
    y_rows = []
    meta_rows = []

    categories = daily["CategoryName"].unique()

    for cat in categories:
        s = daily[daily["CategoryName"] == cat].set_index("OrderDate")[["demand"]]
        s = s.reindex(full_dates, fill_value=0).reset_index().rename(columns={"index": "OrderDate"})
        s["CategoryName"] = cat

        # Need current window + future window
        total_needed = window + horizon_window
        if len(s) < total_needed:
            continue

        # rolling start positions
        for start_i in range(0, len(s) - total_needed + 1, step):
            cur = s.iloc[start_i:start_i + window]["demand"]
            fut = s.iloc[start_i + window:start_i + window + horizon_window]["demand"]

            cur_mean = cur.mean()
            fut_mean = fut.mean()

            # label with threshold
            y = 1 if fut_mean > cur_mean * (1 + increase_threshold) else 0

            # features (current window)
            cur_sum = cur.sum()
            cur_std = cur.std(ddof=0)
            coverage = (cur > 0).mean()
            volatility_ratio = cur_std / (cur_mean + eps)

            # simple trend inside current window: compare first half vs second half
            half = window // 2
            first_half_mean = cur.iloc[:half].mean()
            second_half_mean = cur.iloc[half:].mean()
            intra_growth = (second_half_mean - first_half_mean) / (first_half_mean + eps)

            # future activity indicator (not leaking target, but helps handle all-zero futures)
            # (We keep it OUT to avoid leakage. So we do NOT include future stats as features.)

            X_rows.append({
                "CategoryName": cat,
                "cur_mean": float(cur_mean),
                "cur_sum": float(cur_sum),
                "cur_std": float(cur_std),
                "coverage": float(coverage),
                "volatility_ratio": float(volatility_ratio),
                "intra_growth": float(intra_growth),
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
    data = pd.read_csv(os.path.join(base_dir, "data", "ML-Dataset.csv"))

    output_dir = os.path.join(base_dir, "data", "output")
    os.makedirs(output_dir, exist_ok=True)

    # build dataset
    dataset = build_rolling_features(data)

    # save CSV
    dataset.to_csv(os.path.join(output_dir, "rolling_dataset.csv"), index=False)

    print("Rolling window dataset generated.")
