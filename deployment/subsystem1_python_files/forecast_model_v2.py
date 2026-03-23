from pathlib import Path
import joblib
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DEPLOYMENT_DIR = BASE_DIR.parent
PROJECT_ROOT = DEPLOYMENT_DIR.parent


def resolve_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


ARTIFACTS_PATH = resolve_existing_path(
    PROJECT_ROOT / "models" / "subsystem1_artifacts_v2.pkl",
    DEPLOYMENT_DIR / "models" / "subsystem1_artifacts_v2.pkl",
)
RAW_DATA_PATH = BASE_DIR / "data" / "updated_base_history.csv"
OUTPUT_PATH = BASE_DIR / "data" / "output" / "subsystem1_category_output_v2.csv"

WINDOW = 28
LONG_WINDOW = 90
EPS = 1e-6


def patch_model_compatibility(model) -> None:
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            patch_model_compatibility(step)

    if model.__class__.__name__ == "LogisticRegression" and not hasattr(model, "multi_class"):
        model.multi_class = "auto"


#  build latest snapshot (THIS is what you were missing)
def build_latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data["OrderDate"] = pd.to_datetime(data["OrderDate"], errors="coerce")
    data = data.dropna(subset=["OrderDate"])

    data = data[data["Status"].astype(str).str.lower() == "shipped"].copy()
    data["OrderItemQuantity"] = pd.to_numeric(data["OrderItemQuantity"], errors="coerce").fillna(0)

    if data.empty:
        raise ValueError("No shipped records available after cleaning.")

    latest_date = data["OrderDate"].max()
    start_date = latest_date - pd.Timedelta(days=WINDOW - 1)
    long_start_date = latest_date - pd.Timedelta(days=LONG_WINDOW - 1)

    daily_all = (
        data.groupby(["OrderDate", "CategoryName"])["OrderItemQuantity"]
        .sum()
        .reset_index()
        .rename(columns={"OrderItemQuantity": "demand"})
    )

    all_categories = sorted(data["CategoryName"].dropna().unique().tolist())
    recent_dates = pd.date_range(start_date, latest_date, freq="D")
    long_dates = pd.date_range(long_start_date, latest_date, freq="D")

    rows = []

    for cat in all_categories:
        cat_daily = daily_all[daily_all["CategoryName"] == cat].set_index("OrderDate")[["demand"]]

        recent_series = cat_daily.reindex(recent_dates, fill_value=0)["demand"]
        long_series = cat_daily.reindex(long_dates, fill_value=0)["demand"]

        cur_mean = recent_series.mean()
        cur_sum = recent_series.sum()
        cur_std = recent_series.std(ddof=0)
        coverage = (recent_series > 0).mean()
        volatility_ratio = cur_std / (cur_mean + EPS)

        half = WINDOW // 2
        first_half_mean = recent_series.iloc[:half].mean()
        second_half_mean = recent_series.iloc[half:].mean()
        intra_growth = (second_half_mean - first_half_mean) / (first_half_mean + EPS)

        long_mean = long_series.mean()
        long_sum = long_series.sum()
        long_coverage = (long_series > 0).mean()

        recent_nonzero = np.where(recent_series.to_numpy() > 0)[0]
        if len(recent_nonzero) == 0:
            days_since_last_sale = float(WINDOW)
        else:
            days_since_last_sale = float(WINDOW - 1 - recent_nonzero[-1])

        active_recent_flag = int(cur_sum > 0)
        fallback_baseline = max(float(cur_mean * WINDOW), float(long_mean * WINDOW))

        rows.append({
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
            "window_start": start_date,
            "window_end": latest_date,
        })

    df_feat = pd.DataFrame(rows)
    df_feat["intra_growth"] = df_feat["intra_growth"].clip(-5, 5)
    df_feat["volatility_ratio"] = df_feat["volatility_ratio"].clip(0, 50)
    df_feat["days_since_last_sale"] = df_feat["days_since_last_sale"].clip(0, WINDOW)
    df_feat["month"] = df_feat["window_end"].dt.month
    df_feat["week_of_year"] = df_feat["window_end"].dt.isocalendar().week.astype(int)
    df_feat["day_of_week"] = df_feat["window_end"].dt.dayofweek

    return df_feat


def forecast_demand(df: pd.DataFrame) -> pd.DataFrame:
    artifacts = joblib.load(ARTIFACTS_PATH)
    model = artifacts["model"]
    patch_model_compatibility(model)

    threshold = float(artifacts["threshold"])
    feature_cols = list(artifacts["feature_columns"])
    best_model_name = artifacts.get("best_model_name", "UnknownModel")

    print("Loaded artifact from:", ARTIFACTS_PATH)
    print("Best model name:", best_model_name)
    print("Feature columns:", feature_cols)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[feature_cols]

    probability_up = model.predict_proba(X)[:, 1]
    growth_signal = (probability_up >= threshold).astype(int)

    out = df.copy()
    out["probability_up"] = probability_up
    out["growth_signal"] = growth_signal

    return out


def build_category_output(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data = data.sort_values(["CategoryName", "window_end"])
    latest = data.groupby("CategoryName", as_index=False).tail(1).copy()

    ALPHA = 0.5
    latest["category_base_forecast"] = np.where(
        latest["active_recent_flag"] == 1,
        latest["cur_mean"] * WINDOW,
        latest["fallback_baseline"],
    )
    latest["category_adjusted_forecast"] = np.where(
        latest["active_recent_flag"] == 1,
        latest["category_base_forecast"] * (1 + ALPHA * latest["probability_up"]),
        latest["category_base_forecast"],
    ).round(2)

    keep_cols = [
        "CategoryName",
        "probability_up",
        "growth_signal",
        "active_recent_flag",
        "fallback_baseline",
        "category_base_forecast",
        "category_adjusted_forecast",
        "window_start",
        "window_end",
    ]

    return latest[keep_cols].sort_values("probability_up", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(RAW_DATA_PATH)

    latest_snapshot = build_latest_snapshot(raw_df)
    print(latest_snapshot[[
    "CategoryName",
    "cur_mean",
    "cur_sum",
    "coverage",
    "long_mean",
    "long_sum",
    "days_since_last_sale",
    "active_recent_flag",
    "fallback_baseline",
    "month",
    "week_of_year",
    "day_of_week"
    ]].sort_values("CategoryName"))

    pred_df = forecast_demand(latest_snapshot)
    category_out = build_category_output(pred_df)

    category_out.to_csv(OUTPUT_PATH, index=False)

    print(category_out)
    print("Saved to:", OUTPUT_PATH)
