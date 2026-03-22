from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

UPDATED_BASE_PATH = BASE_DIR / "data" / "updated_base_history.csv"
RAW_DATA_PATH = UPDATED_BASE_PATH if UPDATED_BASE_PATH.exists() else BASE_DIR / "data" / "ML-Dataset.csv"
CATEGORY_FORECAST_INPUT_PATH = BASE_DIR / "data" / "output" / "subsystem1_category_output_v2.csv"
PRODUCT_OUTPUT_PATH = BASE_DIR / "data" / "output" / "subsystem1_product_recommendations_v2.csv"

SHARE_WINDOW_DAYS = 90
MIN_FALLBACK_SHARE = 1e-6


def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH)
    df["OrderDate"] = pd.to_datetime(df["OrderDate"], errors="coerce")

    if "Status" in df.columns:
        df = df[df["Status"].astype(str).str.lower().eq("shipped")]

    required = {"OrderDate", "CategoryName", "ProductName", "OrderItemQuantity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in raw dataset: {missing}")

    df = df.dropna(subset=["OrderDate", "CategoryName", "ProductName"]).copy()
    df["OrderItemQuantity"] = pd.to_numeric(df["OrderItemQuantity"], errors="coerce").fillna(0)
    return df


def load_category_forecast() -> pd.DataFrame:
    df = pd.read_csv(CATEGORY_FORECAST_INPUT_PATH)

    required = {
        "CategoryName",
        "probability_up",
        "growth_signal",
        "active_recent_flag",
        "fallback_baseline",
        "category_base_forecast",
        "category_adjusted_forecast",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in category forecast output: {missing}")

    return df


def compute_product_shares(df: pd.DataFrame, days: int = 90) -> pd.DataFrame:
    latest_date = df["OrderDate"].max().normalize()
    start = latest_date - pd.Timedelta(days=days - 1)

    hist = df[(df["OrderDate"] >= start) & (df["OrderDate"] <= latest_date)].copy()

    if hist.empty:
        raise ValueError("No raw history found in the share window.")

    prod_qty = (
        hist.groupby(["CategoryName", "ProductName"])["OrderItemQuantity"]
        .sum()
        .reset_index()
        .rename(columns={"OrderItemQuantity": "product_qty"})
    )

    cat_qty = (
        prod_qty.groupby("CategoryName")["product_qty"]
        .sum()
        .reset_index()
        .rename(columns={"product_qty": "category_qty"})
    )

    out = prod_qty.merge(cat_qty, on="CategoryName", how="left")
    out["product_share"] = out["product_qty"] / (out["category_qty"] + 1e-9)
    return out


def build_full_product_master(df: pd.DataFrame) -> pd.DataFrame:
    master = (
        df[["CategoryName", "ProductName"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["CategoryName", "ProductName"])
        .reset_index(drop=True)
    )
    return master


def allocate_to_products(raw_df: pd.DataFrame, category_forecast: pd.DataFrame) -> pd.DataFrame:
    shares = compute_product_shares(raw_df, SHARE_WINDOW_DAYS)
    product_master = build_full_product_master(raw_df)

    alloc = product_master.merge(shares, on=["CategoryName", "ProductName"], how="left")

    alloc["product_qty"] = pd.to_numeric(alloc.get("product_qty"), errors="coerce").fillna(0)
    alloc["category_qty"] = pd.to_numeric(alloc.get("category_qty"), errors="coerce").fillna(0)
    alloc["product_share"] = pd.to_numeric(alloc.get("product_share"), errors="coerce").fillna(0)

    alloc = alloc.merge(
        category_forecast[
            [
                "CategoryName",
                "probability_up",
                "growth_signal",
                "active_recent_flag",
                "fallback_baseline",
                "category_base_forecast",
                "category_adjusted_forecast",
            ]
        ],
        on="CategoryName",
        how="left",
    )

    alloc["probability_up"] = pd.to_numeric(alloc["probability_up"], errors="coerce").fillna(0)
    alloc["growth_signal"] = pd.to_numeric(alloc["growth_signal"], errors="coerce").fillna(0).astype(int)
    alloc["active_recent_flag"] = pd.to_numeric(alloc["active_recent_flag"], errors="coerce").fillna(0).astype(int)
    alloc["fallback_baseline"] = pd.to_numeric(alloc["fallback_baseline"], errors="coerce").fillna(0)
    alloc["category_base_forecast"] = pd.to_numeric(alloc["category_base_forecast"], errors="coerce").fillna(0)
    alloc["category_adjusted_forecast"] = pd.to_numeric(alloc["category_adjusted_forecast"], errors="coerce").fillna(0)

    missing_share_mask = alloc["product_share"] <= 0
    category_counts = alloc.groupby("CategoryName")["ProductName"].transform("count").clip(lower=1)
    alloc.loc[missing_share_mask, "product_share"] = 1.0 / category_counts[missing_share_mask]

    alloc["recommended_qty"] = (
        alloc["category_adjusted_forecast"] * alloc["product_share"]
    ).round(0).fillna(0).astype("Int64")

    qty_min = float(alloc["recommended_qty"].min()) if not alloc.empty else 0.0
    qty_max = float(alloc["recommended_qty"].max()) if not alloc.empty else 0.0

    alloc["recommended_qty_norm"] = (
        alloc["recommended_qty"].astype(float) - qty_min
    ) / (qty_max - qty_min + 1e-6)

    alloc["forecast_score"] = (
        0.5 * alloc["probability_up"] +
        0.2 * alloc["product_share"] +
        0.3 * alloc["recommended_qty_norm"]
    ).clip(0, 1)

    alloc["base_reorder_qty"] = alloc["recommended_qty"]
    alloc["confidence_score"] = alloc["probability_up"]
    alloc["share_source"] = np.where(alloc["product_qty"] > 0, "observed_90d", "fallback_equal_split")

    alloc_out = alloc[
        [
            "CategoryName",
            "ProductName",
            "product_share",
            "share_source",
            "product_qty",
            "category_qty",
            "probability_up",
            "growth_signal",
            "active_recent_flag",
            "fallback_baseline",
            "category_base_forecast",
            "category_adjusted_forecast",
            "base_reorder_qty",
            "confidence_score",
            "forecast_score",
        ]
    ].sort_values(
        ["probability_up", "CategoryName", "base_reorder_qty", "product_share"],
        ascending=[False, True, False, False]
    ).reset_index(drop=True)

    return alloc_out


if __name__ == "__main__":
    PRODUCT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_data()
    category_forecast = load_category_forecast()
    alloc_out = allocate_to_products(raw_df, category_forecast)

    alloc_out.to_csv(PRODUCT_OUTPUT_PATH, index=False)
    print("RAW_DATA_PATH:", RAW_DATA_PATH)
    print("CATEGORY_FORECAST_INPUT_PATH:", CATEGORY_FORECAST_INPUT_PATH)
    print(category_forecast.head())
    print(alloc_out.head(20))

    print("Saved:")
    print(" -", PRODUCT_OUTPUT_PATH)