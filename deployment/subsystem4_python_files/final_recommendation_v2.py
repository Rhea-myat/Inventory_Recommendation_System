from pathlib import Path
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent


SUB1_PATH = PROJECT_DIR / "subsystem1_python_files" / "data" / "output" / "subsystem1_product_recommendations_v2.csv"
SUB2_PATH = PROJECT_DIR / "subsystem2_python_files" / "data" / "output" / "category_social_trends_v2.csv"
SUB3_PATH = PROJECT_DIR / "subsystem3_python_files" / "data" / "output" / "risk_scoring_output_v2.csv"

OUTPUT_PATH = PROJECT_DIR / "subsystem4_python_files" / "data" / "output" / "final_recommendations_v2.csv"

# weights
ALPHA = 0.3   # social trend influence
BETA = 0.2    # risk penalty


def load_data():
    sub1 = pd.read_csv(SUB1_PATH)
    sub2 = pd.read_csv(SUB2_PATH)
    sub3 = pd.read_csv(SUB3_PATH)

    for df in [sub1, sub2, sub3]:
        for col in ["base_reorder_qty", "social_trend_score", "risk_score", "probability_up", "forecast_score", "confidence_score"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    print("Subsystem1 rows:", len(sub1))
    print("Subsystem2 rows:", len(sub2))
    print("Subsystem3 rows:", len(sub3))

    return sub1, sub2, sub3


def integrate_signals(sub1: pd.DataFrame, sub2: pd.DataFrame, sub3: pd.DataFrame) -> pd.DataFrame:
    df = sub1.copy()

    sub2_clean = sub2.drop_duplicates(subset=["CategoryName"]).copy()
    sub3_clean = sub3.drop_duplicates(subset=["ProductName"]).copy()

    # category-level social trend
    df = df.merge(
        sub2_clean,
        on="CategoryName",
        how="left"
    )

    # product-level risk
    df = df.merge(
        sub3_clean,
        on="ProductName",
        how="left"
    )

    df["social_trend_score"] = pd.to_numeric(df.get("social_trend_score"), errors="coerce").fillna(0.0)
    df["risk_score"] = pd.to_numeric(df.get("risk_score"), errors="coerce").fillna(0.0)

    return df



def compute_final_recommendation(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["base_reorder_qty"] = pd.to_numeric(out["base_reorder_qty"], errors="coerce").fillna(0)
    out["social_trend_score"] = pd.to_numeric(out["social_trend_score"], errors="coerce").fillna(0)
    out["risk_score"] = pd.to_numeric(out["risk_score"], errors="coerce").fillna(0)

    # social trend increases recommendation, risk acts as penalty
    out["fusion_adjustment"] = (
        ALPHA * out["social_trend_score"]
        - BETA * out["risk_score"]
    )

    out["fusion_adjustment"] = out["fusion_adjustment"].clip(lower=-0.5, upper=1.0)

    out["final_recommended_qty"] = (
        out["base_reorder_qty"] * (1 + out["fusion_adjustment"])
    ).round()

    out["final_recommended_qty"] = out["final_recommended_qty"].clip(lower=0).astype(int)

    return out


def build_output(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "CategoryName",
        "ProductName",
        "base_reorder_qty",
        "social_trend_score",
        "risk_score",
        "fusion_adjustment",
        "final_recommended_qty",
    ]

    optional_cols = [
        "probability_up",
        "growth_signal",
        "active_recent_flag",
        "fallback_baseline",
        "category_base_forecast",
        "category_adjusted_forecast",
        "product_share",
        "share_source",
        "confidence_score",
        "forecast_score",
        "risk_level",
        "risk_drivers",
        "topic_keyword_text",
    ]

    cols = keep_cols + [c for c in optional_cols if c in df.columns]

    output = df[cols].sort_values(
        ["final_recommended_qty", "base_reorder_qty"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return output



if __name__ == "__main__":
    sub1, sub2, sub3 = load_data()

    merged = integrate_signals(sub1, sub2, sub3)
    print("Merged rows:", len(merged))
    final_df = compute_final_recommendation(merged)
    result = build_output(final_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print("Final recommendations generated.")
    print("Saved to:", OUTPUT_PATH)
    print(result.head())