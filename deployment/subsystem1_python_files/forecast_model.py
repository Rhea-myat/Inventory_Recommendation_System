from pathlib import Path
import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEPLOYMENT_DIR = BASE_DIR.parent
PROJECT_ROOT = DEPLOYMENT_DIR.parent


def resolve_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


ARTIFACTS_PATH = resolve_existing_path(
    PROJECT_ROOT / "models" / "subsystem1_artifacts.pkl",
    DEPLOYMENT_DIR / "models" / "subsystem1_artifacts.pkl",
)
INPUT_PATH = BASE_DIR / "data" / "output" / "rolling_dataset.csv"
OUTPUT_PATH = BASE_DIR / "data" / "output" / "subsystem1_inference_output.csv"


def patch_model_compatibility(model) -> None:
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            patch_model_compatibility(step)

    if model.__class__.__name__ == "LogisticRegression" and not hasattr(model, "multi_class"):
        model.multi_class = "auto"


def prepare_inference_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data = data.loc[:, ~data.columns.duplicated()].copy()

    if "cur_sum" in data.columns:
        data = data.drop(columns=["cur_sum"])

    date_cols = ["window_start", "window_end", "future_start", "future_end"]
    for c in date_cols:
        if c in data.columns:
            data[c] = pd.to_datetime(data[c], errors="coerce")

    data = data.sort_values("window_start").reset_index(drop=True)

    data["month"] = data["window_end"].dt.month
    data["week_of_year"] = data["window_end"].dt.isocalendar().week.astype(int)
    data["day_of_week"] = data["window_end"].dt.dayofweek

    return data


def forecast_demand(df: pd.DataFrame) -> pd.DataFrame:
    data = prepare_inference_data(df)

    artifacts = joblib.load(ARTIFACTS_PATH)
    model = artifacts["model"]
    patch_model_compatibility(model)

    threshold = float(artifacts["threshold"])
    feature_cols = list(artifacts["feature_columns"])
    best_model_name = artifacts.get("best_model_name", "UnknownModel")

    missing = [c for c in feature_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns for prediction: {missing}")

    X = data.loc[:, feature_cols]

    probability_up = model.predict_proba(X)[:, 1]
    growth_signal = (probability_up >= threshold).astype(int)

    out = data.copy()
    out["probability_up"] = probability_up
    out["growth_signal"] = growth_signal
    out["model_name"] = best_model_name
    return out


def build_category_output(pred_df: pd.DataFrame) -> pd.DataFrame:
    data = pred_df.copy()
    data = data.sort_values(["CategoryName", "window_end"])
    latest = data.groupby("CategoryName", as_index=False).tail(1).copy()

    keep_cols = [
        "CategoryName",
        "probability_up",
        "growth_signal",
        "window_start",
        "window_end",
    ]
    return latest[keep_cols].sort_values("probability_up", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rolling_df = pd.read_csv(INPUT_PATH)
    pred_df = forecast_demand(rolling_df)
    category_out = build_category_output(pred_df)
    category_out.to_csv(OUTPUT_PATH, index=False)

    print("Forecast category predictions generated.")
    print("Saved to:", OUTPUT_PATH)
