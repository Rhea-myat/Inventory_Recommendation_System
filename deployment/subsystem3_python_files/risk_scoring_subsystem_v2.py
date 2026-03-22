from pathlib import Path
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

UPDATED_BASE_PATH = PROJECT_DIR / "subsystem1_python_files" / "data" / "updated_base_history.csv"
RAW_DATA_PATH = UPDATED_BASE_PATH if UPDATED_BASE_PATH.exists() else PROJECT_DIR / "ML-Dataset.csv"
FORECAST_PATH = PROJECT_DIR / "subsystem1_python_files" / "data" / "output" / "subsystem1_product_recommendations_v2.csv"
OUTPUT_PATH = BASE_DIR / "data" / "output" / "risk_scoring_output_v2.csv"


def classify_risk(score: float) -> str:
    if score >= 0.7:
        return "High"
    elif score >= 0.4:
        return "Medium"
    return "Low"


def risk_drivers(row: pd.Series) -> str:
    drivers = []

    if row["demand_pressure"] > 0.5:
        drivers.append("High forecasted demand")
    if row["availability_risk"] > 0.5:
        drivers.append("Low availability proxy")
    if row["volatility_factor"] > 0.5:
        drivers.append("Unstable demand")
    if row["ops_risk"] > 0.3:
        drivers.append("Operational issues")

    return ", ".join(drivers) if drivers else "No major risk drivers"


def build_risk_output(
    orders_path: Path = RAW_DATA_PATH,
    forecast_path: Path = FORECAST_PATH,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    orders = pd.read_csv(orders_path)
    forecast = pd.read_csv(forecast_path)

    # Aggregate recent orders
    recent_orders = (
        orders.groupby("ProductName")["OrderItemQuantity"]
        .sum()
        .reset_index()
    )

    # Availability proxy (FIXED)
    availability = (         
        orders.groupby("ProductName")["OrderItemQuantity"]
        .mean()
        .reset_index()
        .rename(columns={"OrderItemQuantity": "avg_quantity"})
    )

    # Merge signals
    risk_df = forecast.merge(recent_orders, on="ProductName", how="left")
    risk_df = risk_df.merge(availability, on="ProductName", how="left")

    # Baseline demand
    baseline = (
        orders.groupby("ProductName")["OrderItemQuantity"]
        .mean()
        .reset_index()
    )
    baseline.columns = ["ProductName", "baseline_demand"]
    risk_df = risk_df.merge(baseline, on="ProductName", how="left")

    # Demand pressure
    risk_df["demand_pressure"] = (
        (risk_df["category_adjusted_forecast"] - risk_df["baseline_demand"]) /
        (risk_df["baseline_demand"] + 1e-6)
    )
    risk_df["demand_pressure"] = risk_df["demand_pressure"].clip(0, 1)

    # Availability risk
    risk_df["availability_risk"] = 1 - (
        risk_df["avg_quantity"] /
        (risk_df["avg_quantity"] + risk_df["OrderItemQuantity"] + 1)
    )
    risk_df["availability_risk"] = risk_df["availability_risk"].clip(0, 1)

    # Volatility factor
    volatility = (
        orders.groupby("ProductName")["OrderItemQuantity"]
        .std()
        .reset_index()
    )
    volatility.columns = ["ProductName", "demand_std"]
    volatility["demand_std"] = volatility["demand_std"].fillna(0)

    risk_df = risk_df.merge(volatility, on="ProductName", how="left")

    risk_df["volatility_factor"] = (
        risk_df["demand_std"] / (risk_df["baseline_demand"] + 1)
    )
    risk_df["volatility_factor"] = risk_df["volatility_factor"].clip(0, 1)
    risk_df["volatility_factor"] = risk_df["volatility_factor"].fillna(0)

    # Operational risk
    status_counts = orders.groupby(["ProductName", "Status"]).size().unstack(fill_value=0)
    status_counts["ops_risk"] = (
        status_counts.get("Pending", 0) + status_counts.get("Cancelled", 0)
    ) / status_counts.sum(axis=1)
    status_counts = status_counts.reset_index()[["ProductName", "ops_risk"]]

    risk_df = risk_df.merge(status_counts, on="ProductName", how="left")

    # Fill missing factors
    risk_df["demand_pressure"] = risk_df["demand_pressure"].fillna(0)
    risk_df["availability_risk"] = risk_df["availability_risk"].fillna(0)
    risk_df["ops_risk"] = risk_df["ops_risk"].fillna(0)

    # Weighted risk score
    w1 = 0.35
    w2 = 0.35
    w3 = 0.20
    w4 = 0.10

    risk_df["risk_score"] = (
        w1 * risk_df["demand_pressure"] +
        w2 * risk_df["availability_risk"] +
        w3 * risk_df["volatility_factor"] +
        w4 * risk_df["ops_risk"]
    )

    # Risk labels and drivers
    risk_df["risk_level"] = risk_df["risk_score"].apply(classify_risk)
    risk_df["risk_drivers"] = risk_df.apply(risk_drivers, axis=1)

    risk_output = risk_df[
        [
            "ProductName",
            "demand_pressure",
            "availability_risk",
            "volatility_factor",
            "ops_risk",
            "risk_score",
            "risk_level",
            "risk_drivers"
        ]
    ].copy()

    risk_output.to_csv(output_path, index=False)
    return risk_output


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result = build_risk_output()
    print("Risk scoring output generated.")
    print("Saved to:", OUTPUT_PATH)
    print(result.head())
