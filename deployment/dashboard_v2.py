import pandas as pd
import streamlit as st
from pathlib import Path
import altair as alt
import joblib

st.set_page_config(
    page_title="AI Inventory Recommendation Dashboard",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "subsystem4_python_files" / "data" / "output" / "final_recommendations_v2.csv"
SUB1_ARTIFACTS_PATH = BASE_DIR / "models" / "subsystem1_artifacts_v2.pkl"

# Path to raw history for Subsystem 1 live feature computation
RAW_HISTORY_PATH = BASE_DIR / "subsystem1_python_files" / "data" / "updated_base_history.csv"

SUB1_CANDIDATE_PATHS = [
    BASE_DIR / "subsystem1_python_files" / "data" / "output" / "subsystem1_category_output_v2.csv",
    BASE_DIR / "subsystem1_python_files" / "data" / "output" / "subsystem1_product_recommendations_v2.csv",
]
SUB2_CANDIDATE_PATHS = [
    BASE_DIR / "subsystem2_python_files" / "data" / "output" / "category_social_trends_v2.csv",
    BASE_DIR / "subsystem2_python_files" / "data" / "output" / "social_trend_signal_v2.csv",
]
SUB3_CANDIDATE_PATHS = [
    BASE_DIR / "subsystem3_python_files" / "data" / "output" / "risk_scoring_output_v2.csv"
]


@st.cache_data
def find_existing_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


# New function: load_sub1_artifacts
@st.cache_resource
def load_sub1_artifacts(path: Path) -> dict | None:
    if not path.exists():
        return None

    artifacts = joblib.load(path)
    model = artifacts.get("model")

    if model is not None and hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if step.__class__.__name__ == "LogisticRegression" and not hasattr(step, "multi_class"):
                step.multi_class = "auto"
    elif model is not None and model.__class__.__name__ == "LogisticRegression" and not hasattr(model, "multi_class"):
        model.multi_class = "auto"

    return artifacts


# Helper: Load raw history for Subsystem 1 feature computation
@st.cache_data
def load_raw_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["OrderDate"] = pd.to_datetime(df["OrderDate"], errors="coerce")
    df = df.dropna(subset=["OrderDate"])
    if "Status" in df.columns:
        df = df[df["Status"].astype(str).str.lower() == "shipped"]
    df["OrderItemQuantity"] = pd.to_numeric(df["OrderItemQuantity"], errors="coerce").fillna(0)
    return df


@st.cache_data

def load_data(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    numeric_cols = [
        "base_reorder_qty",
        "social_trend_score",
        "risk_score",
        "final_recommended_qty",
        "probability_up",
        "fusion_adjustment",
        "recommended_qty",
        "category_base_forecast",
        "category_adjusted_forecast",
        "forecast_score",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df



df = load_data(DATA_PATH)

sub1_path = find_existing_path(SUB1_CANDIDATE_PATHS)
sub2_path = find_existing_path(SUB2_CANDIDATE_PATHS)
sub3_path = find_existing_path(SUB3_CANDIDATE_PATHS)

sub1_df = load_data(sub1_path)
sub2_df = load_data(sub2_path)
sub3_df = load_data(sub3_path)

sub1_artifacts = load_sub1_artifacts(SUB1_ARTIFACTS_PATH)

# Load raw history for Subsystem 1 live prediction
raw_history_df = load_raw_history(RAW_HISTORY_PATH)


def build_selection_labels(dataframe: pd.DataFrame) -> list[str]:
    if dataframe.empty:
        return []

    if {"ProductName", "CategoryName"}.issubset(dataframe.columns):
        labels = (
            dataframe["ProductName"].astype(str)
            + " | "
            + dataframe["CategoryName"].astype(str)
        ).drop_duplicates().tolist()
    elif "ProductName" in dataframe.columns:
        labels = dataframe["ProductName"].astype(str).drop_duplicates().tolist()
    elif "CategoryName" in dataframe.columns:
        labels = dataframe["CategoryName"].astype(str).drop_duplicates().tolist()
    else:
        labels = dataframe.index.astype(str).tolist()

    return labels


def get_selected_row(dataframe: pd.DataFrame, selected_label: str) -> pd.Series | None:
    if dataframe.empty:
        return None

    if {"ProductName", "CategoryName"}.issubset(dataframe.columns) and " | " in selected_label:
        product_name, category_name = selected_label.split(" | ", 1)
        selected = dataframe[
            (dataframe["ProductName"].astype(str) == product_name)
            & (dataframe["CategoryName"].astype(str) == category_name)
        ]
        return selected.iloc[0] if not selected.empty else None

    if "ProductName" in dataframe.columns:
        selected = dataframe[dataframe["ProductName"].astype(str) == selected_label]
        return selected.iloc[0] if not selected.empty else None

    if "CategoryName" in dataframe.columns:
        selected = dataframe[dataframe["CategoryName"].astype(str) == selected_label]
        return selected.iloc[0] if not selected.empty else None

    return dataframe.iloc[0] if not dataframe.empty else None


def render_prediction_table(row: pd.Series | None, title: str, preferred_cols: list[str]) -> None:
    st.markdown(f"### {title}")

    if row is None:
        st.info("No prediction available.")
        return

    available_cols = [col for col in preferred_cols if col in row.index]
    if not available_cols:
        st.info("Expected prediction columns were not found in this subsystem output.")
        return

    prediction_df = pd.DataFrame({
        "Field": available_cols,
        "Value": [row[col] for col in available_cols]
    })
    prediction_df.index = prediction_df.index + 1
    st.dataframe(
        prediction_df,
        use_container_width=True,
        hide_index=True,
        height=get_table_height(prediction_df, min_height=120, max_height=260)
    )

def get_table_height(
    df: pd.DataFrame,
    min_height: int = 120,
    max_height: int = 420,
    row_height: int = 35,
    header_height: int = 38,
) -> int:
    row_count = len(df) if df is not None else 0
    calculated_height = header_height + max(row_count, 1) * row_height
    return max(min_height, min(calculated_height, max_height))

# New function: render_sub1_live_prediction (auto-compute features from raw history)
def render_sub1_live_prediction() -> None:
    st.markdown("### Live Subsystem 1 Prediction")
    st.caption("Select a category. The system will automatically compute features and run the model.")

    if sub1_artifacts is None:
        st.warning("Subsystem 1 model artifact file not found.")
        return

    if raw_history_df.empty:
        st.warning("Raw history dataset not found for feature computation.")
        return

    model = sub1_artifacts.get("model")
    threshold = float(sub1_artifacts.get("threshold", 0.5))
    feature_cols = list(sub1_artifacts.get("feature_columns", []))

    categories = sorted(raw_history_df["CategoryName"].dropna().unique().tolist())
    selected_category = st.selectbox("Select Category", categories)

    if st.button("Run Live Prediction"):
        df = raw_history_df.copy()
        df = df[df["CategoryName"] == selected_category]

        if df.empty:
            st.error("No data found for selected category.")
            return

        latest_date = df["OrderDate"].max()
        start_date = latest_date - pd.Timedelta(days=27)
        df_window = df[(df["OrderDate"] >= start_date) & (df["OrderDate"] <= latest_date)]

        daily = (
            df_window.groupby("OrderDate")["OrderItemQuantity"]
            .sum()
            .reindex(pd.date_range(start_date, latest_date), fill_value=0)
        )

        cur = daily
        cur_mean = cur.mean()
        cur_sum = cur.sum()
        cur_std = cur.std()
        coverage = (cur > 0).mean()
        volatility_ratio = cur_std / (cur_mean + 1e-6)

        feature_map = {
            "CategoryName": selected_category,
            "cur_mean": cur_mean,
            "cur_sum": cur_sum,
            "cur_std": cur_std,
            "coverage": coverage,
            "volatility_ratio": volatility_ratio,
            "intra_growth": 0.0,
            "month": latest_date.month,
            "week_of_year": int(latest_date.isocalendar().week),
            "day_of_week": latest_date.dayofweek,
        }

        missing_features = [col for col in feature_cols if col not in feature_map]
        if missing_features:
            st.error(f"Missing required model features: {missing_features}")
            return
        input_df = pd.DataFrame([{col: feature_map[col] for col in feature_cols}])

        probability_up = float(model.predict_proba(input_df)[0, 1])
        growth_signal = int(probability_up >= threshold)

        base_forecast = cur_mean * 28
        adjusted_forecast = base_forecast * (1 + 0.5 * probability_up)

        category_result = pd.DataFrame([
            {"Field": "Category", "Value": selected_category},
            {"Field": "probability_up", "Value": round(probability_up, 4)},
            {"Field": "growth_signal", "Value": growth_signal},
            {"Field": "category_base_forecast", "Value": round(base_forecast, 2)},
            {"Field": "category_adjusted_forecast", "Value": round(adjusted_forecast, 2)},
        ])

        st.success("Prediction completed using model + computed features.")
        computed_features_df = pd.DataFrame([
            {"Feature": col, "Value": feature_map[col]} for col in feature_cols
        ])

        st.markdown("#### Category-Level Output")
        st.dataframe(
            category_result,
            use_container_width=True,
            hide_index=True,
            height=get_table_height(category_result, min_height=120, max_height=220)
        )

        st.markdown("#### Computed Features Used by the Model")
        st.dataframe(
            computed_features_df,
            use_container_width=True,
            hide_index=True,
            height=get_table_height(computed_features_df, min_height=120, max_height=320)
        )

        st.markdown("#### Product-Level Allocation Output")

        latest_history_date = raw_history_df["OrderDate"].max().normalize()
        share_start_date = latest_history_date - pd.Timedelta(days=89)
        share_df = raw_history_df[
            (raw_history_df["OrderDate"] >= share_start_date)
            & (raw_history_df["OrderDate"] <= latest_history_date)
            & (raw_history_df["CategoryName"] == selected_category)
        ].copy()

        if share_df.empty:
            st.info("No recent product-level history found for the selected category.")
        else:
            product_qty_df = (
                share_df.groupby(["CategoryName", "ProductName"], as_index=False)["OrderItemQuantity"]
                .sum()
                .rename(columns={"OrderItemQuantity": "product_qty"})
            )

            category_qty = product_qty_df["product_qty"].sum()
            product_qty_df["product_share"] = product_qty_df["product_qty"] / (category_qty + 1e-9)

            product_qty_df["recommended_qty"] = (
                adjusted_forecast * product_qty_df["product_share"]
            ).round(0).astype(int)

            qty_min = product_qty_df["recommended_qty"].min()
            qty_max = product_qty_df["recommended_qty"].max()
            product_qty_df["recommended_qty_norm"] = (
                product_qty_df["recommended_qty"] - qty_min
            ) / (qty_max - qty_min + 1e-6)

            product_qty_df["forecast_score"] = (
                0.5 * probability_up
                + 0.2 * product_qty_df["product_share"]
                + 0.3 * product_qty_df["recommended_qty_norm"]
            ).clip(0, 1)

            product_qty_df["base_reorder_qty"] = product_qty_df["recommended_qty"]
            product_qty_df["confidence_score"] = probability_up
            product_qty_df["category_base_forecast"] = round(base_forecast, 2)
            product_qty_df["category_adjusted_forecast"] = round(adjusted_forecast, 2)
            product_qty_df["probability_up"] = round(probability_up, 4)
            product_qty_df["growth_signal"] = growth_signal

            product_output_df = product_qty_df[[
                "CategoryName",
                "ProductName",
                "product_qty",
                "product_share",
                "probability_up",
                "growth_signal",
                "category_base_forecast",
                "category_adjusted_forecast",
                "base_reorder_qty",
                "confidence_score",
                "forecast_score",
            ]].sort_values(["base_reorder_qty", "product_share"], ascending=[False, False]).reset_index(drop=True)

            product_output_df.index = product_output_df.index + 1
            st.dataframe(
                product_output_df,
                use_container_width=True,
                height=get_table_height(product_output_df, min_height=120, max_height=320)
            )

st.title("AI Inventory Recommendation Dashboard")
# today date
st.caption(f"Data last updated: {pd.to_datetime('today').strftime('%Y-%m-%d')}")
#next 28days

st.caption(f"Recommendation for {pd.to_datetime('today').strftime('%Y-%m-%d')} to {(pd.to_datetime('today') + pd.Timedelta(days=28)).strftime('%Y-%m-%d')}")

dashboard_tab, sub1_tab, sub2_tab, sub3_tab, sub4_tab = st.tabs([
    "Dashboard Overview",
    "Subsystem 1 Test",
    "Subsystem 2 Test",
    "Subsystem 3 Test",
    "Subsystem 4 Test",
])

with dashboard_tab:
# ---------- Sidebar filters ----------
    st.sidebar.header("Filters")

    categories = ["All"] + sorted(df["CategoryName"].dropna().unique().tolist()) if "CategoryName" in df.columns else ["All"]
    selected_category = st.sidebar.selectbox("Category", categories)

    risk_levels = ["All"]
    if "risk_level" in df.columns:
        risk_levels += sorted(df["risk_level"].dropna().unique().tolist())
    selected_risk = st.sidebar.selectbox("Risk Level", risk_levels)

    filtered_df = df.copy()

    max_products = len(filtered_df)

    top_n = st.sidebar.slider(
        "Top N products",
        min_value=5,
        max_value=max_products,
        value=min(10, max_products)
    )



    if selected_category != "All":
        filtered_df = filtered_df[filtered_df["CategoryName"] == selected_category]

    if selected_risk != "All" and "risk_level" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["risk_level"] == selected_risk]

    # ---------- KPIs ----------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Products", len(filtered_df))

    with col2:
        if "final_recommended_qty" in filtered_df.columns:
            st.metric("Total Final Recommended Qty", int(filtered_df["final_recommended_qty"].fillna(0).sum()))
        else:
            st.metric("Total Final Recommended Qty", "N/A")

    with col3:
        if "base_reorder_qty" in filtered_df.columns:
            st.metric("Total Base Reorder Qty", int(filtered_df["base_reorder_qty"].fillna(0).sum()))
        else:
            st.metric("Total Base Reorder Qty", "N/A")

    with col4:
        if "risk_score" in filtered_df.columns and not filtered_df["risk_score"].dropna().empty:
            st.metric("Average Risk Score", round(filtered_df["risk_score"].mean(), 3))
        else:
            st.metric("Average Risk Score", "N/A")

    # ---------- Main table ----------
    st.subheader("Final Recommendations")

    display_cols = [
        col for col in [
            "CategoryName",
            "ProductName",
            "base_reorder_qty",
            "social_trend_score",
            "risk_score",
            "risk_level",
            "probability_up",
            "fusion_adjustment",
            "final_recommended_qty",
            "risk_drivers",
            "topic_keyword_text",
        ] if col in filtered_df.columns
    ]

    table_df = filtered_df[display_cols].reset_index(drop=True)
    table_df.index = table_df.index + 1

    st.dataframe(
        table_df,
        use_container_width=True,
        height=get_table_height(table_df, min_height=140, max_height=420)
    )

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download filtered recommendations as CSV",
        data=csv_data,
        file_name="filtered_recommendations.csv",
        mime="text/csv"
    )

    # ---------- Top recommended ----------
    st.subheader("Top Recommended Products")

    top_df = filtered_df.copy()
    if "final_recommended_qty" in top_df.columns:
        top_df = top_df.sort_values("final_recommended_qty", ascending=False).head(top_n)

    top_display_cols = [
        col for col in [
            "CategoryName",
            "ProductName",
            "base_reorder_qty",
            "final_recommended_qty",
            "risk_level",
            "social_trend_score",
        ] if col in top_df.columns
    ]

    top_df = top_df[top_display_cols].reset_index(drop=True)
    top_df.index = top_df.index + 1
    st.dataframe(top_df, use_container_width=True)

    # ---------- Charts ----------
    # ---------- Charts ----------
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Top Products by Final Recommended Qty")
        if "final_recommended_qty" in filtered_df.columns and "ProductName" in filtered_df.columns:
            chart_df = (
                filtered_df[["ProductName", "final_recommended_qty"]]
                .dropna()
                .copy()
            )

            chart_df["ProductName"] = chart_df["ProductName"].astype(str)
            chart_df["final_recommended_qty"] = pd.to_numeric(
                chart_df["final_recommended_qty"], errors="coerce"
            )

            chart_df = (
                chart_df.sort_values("final_recommended_qty", ascending=False)
                .head(top_n)
                .reset_index(drop=True)
            )

            bar1 = alt.Chart(chart_df).mark_bar().encode(
                x=alt.X("ProductName:N", sort="-y", title="Product"),
                y=alt.Y("final_recommended_qty:Q", title="Final Recommended Qty"),
                tooltip=[
                    alt.Tooltip("ProductName:N", title="Product"),
                    alt.Tooltip("final_recommended_qty:Q", title="Final Recommended Qty")
                ]
            ).properties(
                height=320
            )

            st.altair_chart(bar1, use_container_width=True)
        else:
            st.info("Required columns not found for this chart.")

    with chart_col2:
        st.subheader("Risk Level Distribution")
        if "risk_level" in filtered_df.columns:
            risk_counts = (
                filtered_df["risk_level"]
                .dropna()
                .value_counts()
                .reset_index()
            )
            risk_counts.columns = ["risk_level", "count"]

            risk_counts["risk_level"] = risk_counts["risk_level"].astype(str)
            risk_counts["count"] = pd.to_numeric(risk_counts["count"], errors="coerce")

            bar2 = alt.Chart(risk_counts).mark_bar().encode(
                x=alt.X("risk_level:N", title="Risk Level"),
                y=alt.Y("count:Q", title="Count"),
                tooltip=[
                    alt.Tooltip("risk_level:N", title="Risk Level"),
                    alt.Tooltip("count:Q", title="Count")
                ]
            ).properties(
                height=320
            )

            st.altair_chart(bar2, use_container_width=True)
        else:
            st.info("Risk level column not found.")

    # ---------- Category summary ----------
    st.subheader("Category Summary")

    if {"CategoryName", "final_recommended_qty"}.issubset(filtered_df.columns):
        summary_dict = {
            "final_recommended_qty": "sum"
        }

        if "base_reorder_qty" in filtered_df.columns:
            summary_dict["base_reorder_qty"] = "sum"
        if "risk_score" in filtered_df.columns:
            summary_dict["risk_score"] = "mean"
        if "social_trend_score" in filtered_df.columns:
            summary_dict["social_trend_score"] = "mean"

        category_summary = (
            filtered_df.groupby("CategoryName", as_index=False)
            .agg(summary_dict)
            .sort_values("final_recommended_qty", ascending=False)
            .reset_index(drop=True)
        )

        # start index from 1
        category_summary.index = category_summary.index + 1

        st.dataframe(category_summary, use_container_width=True)

    else:
        st.info("Category summary cannot be generated from current columns.")

    # ---------- Feedback Section ----------
    st.subheader("Feedback Collection")

    required_feedback_cols = ["ProductName", "final_recommended_qty"]

    if all(col in filtered_df.columns for col in required_feedback_cols) and not filtered_df.empty:

        feedback_options_df = filtered_df.copy()

        # Build display label for product selection
        if "CategoryName" in feedback_options_df.columns:
            feedback_options_df["feedback_label"] = (
                feedback_options_df["ProductName"].astype(str)
                + " | "
                + feedback_options_df["CategoryName"].astype(str)
            )
        else:
            feedback_options_df["feedback_label"] = feedback_options_df["ProductName"].astype(str)

        selected_feedback_label = st.selectbox(
            "Select a product to submit feedback",
            feedback_options_df["feedback_label"].tolist()
        )

        selected_row = feedback_options_df[
            feedback_options_df["feedback_label"] == selected_feedback_label
        ].iloc[0]

        product_name = selected_row["ProductName"]
        recommended_qty = selected_row["final_recommended_qty"]

        trend_score = selected_row["social_trend_score"] if "social_trend_score" in selected_row.index else None
        risk_score = selected_row["risk_score"] if "risk_score" in selected_row.index else None

        st.write(f"**Selected Product:** {product_name}")
        st.write(f"**System Recommended Quantity:** {int(recommended_qty) if pd.notna(recommended_qty) else 'N/A'}")

        final_qty = st.number_input(
            "Final Ordered Quantity",
            min_value=0,
            value=int(recommended_qty) if pd.notna(recommended_qty) else 0,
            step=1
        )

        actual_sales = st.number_input(
            "Actual Sales (after period)",
            min_value=0,
            value=0,
            step=1
        )

        stockout_flag = st.selectbox("Stockout occurred?", ["No", "Yes"])
        overstock_flag = st.selectbox("Overstock occurred?", ["No", "Yes"])

        if st.button("Submit Feedback"):

            feedback = {
                "date": pd.Timestamp.now(),
                "product_name": product_name,
                "category_name": selected_row["CategoryName"] if "CategoryName" in selected_row.index else None,
                "recommended_qty": recommended_qty,
                "final_ordered_qty": final_qty,
                "override_flag": int(final_qty != recommended_qty),
                "actual_sales": actual_sales,
                "stockout_flag": int(stockout_flag == "Yes"),
                "overstock_flag": int(overstock_flag == "Yes"),
                "trend_score": trend_score,
                "risk_score": risk_score
            }

            feedback_df = pd.DataFrame([feedback])

            feedback_dir = BASE_DIR / "feedback"
            feedback_dir.mkdir(parents=True, exist_ok=True)
            file_path = feedback_dir / "feedback_log.csv"

            if file_path.exists():
                feedback_df.to_csv(file_path, mode="a", header=False, index=False)
            else:
                feedback_df.to_csv(file_path, index=False)

            st.success("Feedback saved!")

    else:
        st.info("No products available for feedback collection.")


with sub1_tab:
    st.subheader("Subsystem 1 Interactive Testing")
    st.caption("Select a product or category and view the forecasting output generated by Subsystem 1.")

    if sub1_df.empty:
        st.warning("Subsystem 1 output file not found. Update the candidate file paths in the dashboard if your file name is different.")
    else:
        sub1_labels = build_selection_labels(sub1_df)
        selected_sub1 = st.selectbox(
            "Choose a Subsystem 1 input",
            sub1_labels,
            key="sub1_selector",
        )
        sub1_row = get_selected_row(sub1_df, selected_sub1)

        render_prediction_table(
            sub1_row,
            "Subsystem 1 Prediction Output",
            [
                "CategoryName",
                "ProductName",
                "probability_up",
                "recommended_qty",
                "base_reorder_qty",
                "category_base_forecast",
                "category_adjusted_forecast",
                "forecast_score",
            ],
        )

        st.markdown("### Subsystem 1 Raw Output Preview")
        preview_df = sub1_df.reset_index(drop=True).copy()
        preview_df.index = preview_df.index + 1
        st.dataframe(
            preview_df,
            use_container_width=True,
            height=get_table_height(preview_df, min_height=120, max_height=320)
        )

    st.divider()
    render_sub1_live_prediction()

with sub2_tab:
    st.subheader("Subsystem 2 Interactive Testing")
    st.caption("Select a product or category and view the social trend output generated by Subsystem 2.")

    if sub2_df.empty:
        st.warning("Subsystem 2 output file not found. Update the candidate file paths in the dashboard if your file name is different.")
    else:
        sub2_labels = build_selection_labels(sub2_df)
        selected_sub2 = st.selectbox(
            "Choose a Subsystem 2 input",
            sub2_labels,
            key="sub2_selector",
        )
        sub2_row = get_selected_row(sub2_df, selected_sub2)

        render_prediction_table(
            sub2_row,
            "Subsystem 2 Prediction Output",
            [
                "CategoryName",
                "ProductName",
                "social_trend_score",
                "topic_keyword_text",
                "trend_label",
                "sentiment_label",
                "sentiment_score",
            ],
        )

        st.markdown("### Subsystem 2 Raw Output Preview")
        preview_df = sub2_df.reset_index(drop=True).copy()
        preview_df.index = preview_df.index + 1
        st.dataframe(
            preview_df,
            use_container_width=True,
            height=get_table_height(preview_df, min_height=120, max_height=320)
        )

with sub3_tab:
    st.subheader("Subsystem 3 Interactive Testing")
    st.caption("Select a product or category and view the risk scoring output generated by Subsystem 3.")

    if sub3_df.empty:
        st.warning("Subsystem 3 output file not found. Update the candidate file paths in the dashboard if your file name is different.")
    else:
        sub3_labels = build_selection_labels(sub3_df)
        selected_sub3 = st.selectbox(
            "Choose a Subsystem 3 input",
            sub3_labels,
            key="sub3_selector",
        )
        sub3_row = get_selected_row(sub3_df, selected_sub3)

        render_prediction_table(
            sub3_row,
            "Subsystem 3 Prediction Output",
            [
                "CategoryName",
                "ProductName",
                "risk_score",
                "risk_level",
                "risk_drivers",
                "base_reorder_qty",
            ],
        )

        st.markdown("### Subsystem 3 Raw Output Preview")
        preview_df = sub3_df.reset_index(drop=True).copy()
        preview_df.index = preview_df.index + 1
        st.dataframe(
            preview_df,
            use_container_width=True,
            height=get_table_height(preview_df, min_height=120, max_height=320)
        )

with sub4_tab:
    st.subheader("Subsystem 4 Interactive Testing")
    st.caption("Select a product or category and view the fused final recommendation generated by Subsystem 4.")

    if df.empty:
        st.warning("Subsystem 4 output file not found.")
    else:
        sub4_labels = build_selection_labels(df)
        selected_sub4 = st.selectbox(
            "Choose a Subsystem 4 input",
            sub4_labels,
            key="sub4_selector",
        )
        sub4_row = get_selected_row(df, selected_sub4)

        render_prediction_table(
            sub4_row,
            "Subsystem 4 Prediction Output",
            [
                "CategoryName",
                "ProductName",
                "base_reorder_qty",
                "social_trend_score",
                "risk_score",
                "risk_level",
                "probability_up",
                "fusion_adjustment",
                "final_recommended_qty",
                "risk_drivers",
                "topic_keyword_text",
            ],
        )

        st.markdown("### Subsystem 4 Raw Output Preview")
        preview_df = df.reset_index(drop=True).copy()
        preview_df.index = preview_df.index + 1
        st.dataframe(
            preview_df,
            use_container_width=True,
            height=get_table_height(preview_df, min_height=120, max_height=320)
        )