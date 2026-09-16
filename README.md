# AI Inventory Recommendation System

An end-to-end inventory decision-support application that combines demand forecasting, product allocation, social-trend analysis, operational risk scoring, and final reorder recommendations.

The project includes a Streamlit dashboard, reproducible Docker environment, saved model artifacts, subsystem testing views, recommendation export, and a feedback workflow for comparing system recommendations with final ordering decisions.

## Live Demo 

Try the deployed application on Streamlit Community Cloud: [Launch the Inventory Recommendation System](https://myat-rhea-inventory-recommender.streamlit.app/)

## Project Overview

Inventory planning requires more than historical demand alone. Product demand may also be affected by emerging trends, stock availability, volatility, and operational issues.

This system combines four analytical subsystems:

1. Demand forecasting and product allocation
2. Social-trend detection
3. Inventory and operational risk scoring
4. Final recommendation generation

The resulting dashboard helps users review recommended reorder quantities, compare products and categories, filter by risk level, export recommendations, and record decision outcomes.

## Main Features

- Demand-direction forecasting from historical order data
- Category-level demand estimation
- Product-level quantity allocation
- Social-trend detection using text embeddings and topic clustering
- Matching of emerging topics to product categories
- Inventory and operational risk scoring
- Identification of major risk drivers
- Fusion of demand, trend, and risk signals
- Final recommended reorder quantities
- Category and risk-level filters
- Interactive charts and KPI summaries
- CSV export of filtered recommendations
- Feedback collection for ordered quantity, sales, stockouts, and overstock
- Interactive testing views for all four subsystems
- Saved model artifacts and execution logs
- Docker configuration for a consistent runtime environment

## System Architecture

```text
Historical Orders
       |
       v
Subsystem 1
Demand Forecasting and Product Allocation
       |
       +------------------+
       |                  |
       v                  v
Subsystem 2          Subsystem 3
Social-Trend         Risk Scoring
Detection            and Risk Drivers
       |                  |
       +--------+---------+
                |
                v
Subsystem 4
Recommendation Fusion
                |
                v
Final Reorder Recommendations
                |
                v
Streamlit Dashboard
       |
       +----------------------+
       |                      |
       v                      v
CSV Export              User Feedback
```

## Analytical Subsystems

### Subsystem 1: Demand Forecasting and Allocation

Subsystem 1 builds rolling demand features from historical order data, estimates demand direction, produces category-level forecasts, and allocates recommended quantities to individual products.

Key outputs include:

- Category base forecast
- Category adjusted forecast
- Product demand share
- Base reorder quantity
- Forecast score
- Demand probability

### Subsystem 2: Social-Trend Detection

Subsystem 2 cleans social-media text, creates sentence embeddings, clusters related topics, calculates trend scores, and matches trending topics to inventory categories.

The live demonstration uses the `all-MiniLM-L6-v2` sentence-transformer model to compare user-entered text with saved topic centroids.

Key outputs include:

- Topic keywords
- Trend score
- Trending status
- Matched product category
- Category social-trend score

### Subsystem 3: Risk Scoring

Subsystem 3 combines several inventory and operational signals:

- Demand pressure
- Product availability
- Demand volatility
- Operational status

The subsystem produces:

- Composite risk score
- Risk level
- Human-readable risk drivers

### Subsystem 4: Final Recommendation Engine

Subsystem 4 combines the outputs of the first three subsystems.

The final quantity starts with the base reorder recommendation and is adjusted using social-trend and risk signals.

Key outputs include:

- Base reorder quantity
- Social-trend score
- Risk score and risk level
- Fusion adjustment
- Final recommended quantity
- Recommendation explanation

## Dashboard

The Streamlit dashboard provides:

- Product, category, and risk-level filtering
- Recommendation KPIs
- Final recommendation table
- Top-product rankings
- Risk-level distribution
- Category summaries
- Downloadable CSV recommendations
- Interactive subsystem demonstrations
- Feedback collection for post-recommendation evaluation

## Technologies

- Python 3.11
- Streamlit
- pandas
- NumPy
- scikit-learn
- Sentence Transformers
- HDBSCAN
- Altair
- Plotly
- Matplotlib
- Seaborn
- Docker
- Jupyter Notebook

## Repository Structure

```text
.
├── deployment/
│   ├── dashboard_v2.py
│   ├── main_system_v2.py
│   ├── subsystem1_python_files/
│   ├── subsystem2_python_files/
│   ├── subsystem3_python_files/
│   └── subsystem4_python_files/
├── development/
│   ├── subsystem1/
│   ├── subsystem2/
│   ├── subsystem3/
│   └── subsystem4/
├── feedback/
├── model_logs/
├── models/
├── Dockerfile
├── ML-Dataset.csv
├── requirements.txt
└── updated_base_history.csv
```

## Feedback Workflow

The dashboard allows users to record:

- System-recommended quantity
- Final ordered quantity
- Actual sales
- Whether a stockout occurred
- Whether overstock occurred
- Associated trend and risk scores

This information is stored in:

```text
feedback/feedback_log.csv
```

The feedback structure can support future evaluation of recommendation quality and human overrides.


