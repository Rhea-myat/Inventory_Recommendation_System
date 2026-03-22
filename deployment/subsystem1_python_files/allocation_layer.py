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


def allocate_to_products(raw_df: pd.DataFrame, category_forecast: pd.DataFrame) -> pd.DataFrame:
    shares = compute_product_shares(raw_df, SHARE_WINDOW_DAYS)

    alloc = shares.merge(
        category_forecast[
            [
                "CategoryName",
                "probability_up",
                "growth_signal",
                "category_base_forecast",
                "category_adjusted_forecast",
            ]
        ],
        on="CategoryName",
        how="inner",
    )

    alloc["recommended_qty"] = (
        alloc["category_adjusted_forecast"] * alloc["product_share"]
    ).round(0).fillna(0).astype("Int64")

    qty_min = alloc["recommended_qty"].min()
    qty_max = alloc["recommended_qty"].max()

    alloc["recommended_qty_norm"] = (
        alloc["recommended_qty"] - qty_min
    ) / (qty_max - qty_min + 1e-6)

    alloc["forecast_score"] = (
        0.5 * alloc["probability_up"] +
        0.2 * alloc["product_share"] +
        0.3 * alloc["recommended_qty_norm"]
    ).clip(0, 1)

    alloc["base_reorder_qty"] = alloc["recommended_qty"]
    alloc["confidence_score"] = alloc["probability_up"]

    alloc_out = alloc[
        [
            "CategoryName",
            "ProductName",
            "product_share",
            "product_qty",
            "category_qty",
            "probability_up",
            "growth_signal",
            "category_base_forecast",
            "category_adjusted_forecast",
            "base_reorder_qty",
            "confidence_score",
            "forecast_score",
        ]
    ].sort_values(["probability_up", "CategoryName", "product_share"], ascending=[False, True, False])

    return alloc_out


if __name__ == "__main__":
    PRODUCT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_data()
    category_forecast = load_category_forecast()
    alloc_out = allocate_to_products(raw_df, category_forecast)

    alloc_out.to_csv(PRODUCT_OUTPUT_PATH, index=False)

    print("Saved:")
    print(" -", PRODUCT_OUTPUT_PATH)