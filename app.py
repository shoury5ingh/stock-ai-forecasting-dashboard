
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------
# Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Stock AI Research Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports" / "final_test_evaluation"

SUMMARY_FILE = REPORTS / "final_test_summary.csv"
METRICS_FILE = REPORTS / "final_test_metrics.csv"
PREDICTIONS_FILE = REPORTS / "final_test_predictions.csv"

MODEL_LABELS = {
    "Zero_Return": "Zero Return",
    "Historical_Mean": "Historical Mean",
    "Historical_Median": "Historical Median",
    "Previous_Day_Return": "Previous Day Return",
    "Linear_Regression": "Linear Regression",
    "Ridge_Regression": "Ridge Regression",
    "Random_Forest": "Random Forest",
    "XGBoost": "XGBoost",
    "LSTM": "LSTM",
}

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.15rem;
        font-weight: 750;
        margin-bottom: 0.1rem;
    }
    .subtitle {
        color: #9ca3af;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .notice {
        padding: 0.9rem 1rem;
        border-radius: 0.5rem;
        background: rgba(128,128,128,0.12);
        border-left: 4px solid #64748b;
        margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Load data
# --------------------------------------------------
@st.cache_data
def load_data():
    summary = pd.read_csv(SUMMARY_FILE)
    metrics = pd.read_csv(METRICS_FILE)
    predictions = pd.read_csv(PREDICTIONS_FILE, parse_dates=["Date"])

    summary["Model_Label"] = summary["Model"].map(MODEL_LABELS).fillna(
        summary["Model"]
    )
    metrics["Model_Label"] = metrics["Model"].map(MODEL_LABELS).fillna(
        metrics["Model"]
    )

    predictions["Date"] = pd.to_datetime(predictions["Date"])

    return summary, metrics, predictions


st.markdown(
    '<div class="main-title">📈 Cross-Market Stock Return Forecasting</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Machine Learning • Deep Learning • Time-Series Evaluation</div>',
    unsafe_allow_html=True,
)

if not all(
    path.exists()
    for path in [SUMMARY_FILE, METRICS_FILE, PREDICTIONS_FILE]
):
    st.error(
        "Evaluation files are missing. Please check "
        "reports/final_test_evaluation/."
    )
    st.stop()

try:
    summary, metrics, predictions = load_data()
except Exception as exc:
    st.error(f"Unable to load evaluation data: {exc}")
    st.stop()

if summary.empty or metrics.empty or predictions.empty:
    st.error("One or more evaluation files are empty.")
    st.stop()

st.markdown(
    """
    <div class="notice">
    <strong>Research disclaimer:</strong> This project evaluates historical
    out-of-sample return predictions. It is not a live trading system,
    investment recommendation, or guaranteed forecasting tool.
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("Dashboard Controls")

tickers = sorted(metrics["Ticker"].dropna().unique().tolist())

selected_ticker = st.sidebar.selectbox(
    "Select stock",
    tickers,
    index=0,
)

ticker_metrics = metrics[metrics["Ticker"] == selected_ticker].copy()
ticker_predictions = predictions[
    predictions["Ticker"] == selected_ticker
].copy()

ticker_predictions = ticker_predictions.sort_values("Date")

selected_market = (
    "Indian Market"
    if selected_ticker in [
        "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY"
    ]
    else "US Market"
)

st.sidebar.markdown("---")
st.sidebar.write(f"**Market:** {selected_market}")
st.sidebar.write(f"**Stock:** {selected_ticker}")
st.sidebar.write(f"**Test observations:** {len(ticker_predictions):,}")

model_options = [
    col for col in ticker_predictions.columns
    if col not in ["Date", "Ticker", "Actual_Return"]
]

selected_model = st.sidebar.selectbox(
    "Prediction model",
    model_options,
    index=model_options.index("LSTM") if "LSTM" in model_options else 0,
)


# --------------------------------------------------
# Overview metrics
# --------------------------------------------------
best_row = ticker_metrics.loc[ticker_metrics["MAE"].idxmin()]
overall_best = summary.loc[summary["Average_MAE"].idxmin()]

test_start = ticker_metrics["Test_Start"].iloc[0]
test_end = ticker_metrics["Test_End"].iloc[0]
test_rows = int(ticker_metrics["Test_Rows"].iloc[0])

st.subheader("Selected Stock Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Market", selected_market)
c2.metric("Test observations", f"{test_rows:,}")
c3.metric("Best model (MAE)", MODEL_LABELS.get(best_row["Model"], best_row["Model"]))
c4.metric("Best model MAE", f'{best_row["MAE"]:.4f}')

st.caption(
    f"Test period: {test_start} to {test_end} | "
    f"Selected stock: {selected_ticker}"
)

st.divider()


# --------------------------------------------------
# Main tabs
# --------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Model Leaderboard",
        "📈 Prediction Analysis",
        "🔬 Stock Evaluation",
        "📘 Methodology & Findings",
    ]
)


# --------------------------------------------------
# Tab 1: Leaderboard
# --------------------------------------------------
with tab1:
    st.subheader("Overall Model Performance")

    st.write(
        "Average performance across all 10 stocks on the untouched "
        "final test period."
    )

    display_summary = summary.sort_values("Average_MAE").copy()

    display_summary["Model"] = display_summary["Model_Label"]

    display_summary = display_summary[
        [
            "Model",
            "Average_MAE",
            "Average_RMSE",
            "Average_R2",
            "Average_Directional_Accuracy",
        ]
    ]

    st.dataframe(
        display_summary.style.format(
            {
                "Average_MAE": "{:.4f}",
                "Average_RMSE": "{:.4f}",
                "Average_R2": "{:.4f}",
                "Average_Directional_Accuracy": "{:.2f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    fig = px.bar(
        display_summary.sort_values("Average_MAE"),
        x="Average_MAE",
        y="Model",
        orientation="h",
        title="Average MAE — Lower is Better",
        text_auto=".4f",
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        xaxis_title="Mean Absolute Error",
        yaxis_title="",
        height=480,
    )
    st.plotly_chart(fig, use_container_width=True)

    best = summary.loc[summary["Average_MAE"].idxmin()]

    st.info(
        f"Overall lowest average MAE: "
        f"{MODEL_LABELS.get(best['Model'], best['Model'])} "
        f"({best['Average_MAE']:.4f})."
    )


# --------------------------------------------------
# Tab 2: Actual vs predicted
# --------------------------------------------------
with tab2:
    st.subheader(f"Actual vs Predicted Returns — {selected_ticker}")

    st.write(
        "This chart displays historical predictions from the final test "
        "period, not a forecast for the next trading session."
    )

    if selected_model not in ticker_predictions.columns:
        st.warning("The selected model prediction column is unavailable.")
    else:
        chart_data = ticker_predictions[
            ["Date", "Actual_Return", selected_model]
        ].copy()

        chart_data = chart_data.rename(
            columns={
                "Actual_Return": "Actual Return",
                selected_model: "Predicted Return",
            }
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=chart_data["Date"],
                y=chart_data["Actual Return"],
                mode="lines",
                name="Actual Return",
                line=dict(width=1.3),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=chart_data["Date"],
                y=chart_data["Predicted Return"],
                mode="lines",
                name="Predicted Return",
                line=dict(width=1.5),
            )
        )

        fig.update_layout(
            title=f"{selected_ticker}: Actual vs {MODEL_LABELS.get(selected_model, selected_model)}",
            xaxis_title="Date",
            yaxis_title="Return (dataset units)",
            hovermode="x unified",
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "Download selected stock predictions (CSV)",
            data=ticker_predictions.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_ticker}_test_predictions.csv",
            mime="text/csv",
        )


# --------------------------------------------------
# Tab 3: Stock evaluation
# --------------------------------------------------
with tab3:
    st.subheader(f"Model Comparison — {selected_ticker}")

    sorted_metrics = ticker_metrics.sort_values("MAE").copy()

    fig = px.bar(
        sorted_metrics,
        x="Model_Label",
        y="MAE",
        color="Model_Label",
        title="MAE by Model",
        text_auto=".4f",
    )
    fig.update_layout(
        xaxis_title="Model",
        yaxis_title="Mean Absolute Error",
        showlegend=False,
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.bar(
        sorted_metrics,
        x="Model_Label",
        y="Directional_Accuracy",
        title="Directional Accuracy by Model",
        text_auto=".2f",
    )
    fig2.add_hline(
        y=50,
        line_dash="dash",
        annotation_text="50% reference",
    )
    fig2.update_layout(
        xaxis_title="Model",
        yaxis_title="Directional Accuracy (%)",
        height=400,
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Detailed metrics")

    st.dataframe(
        sorted_metrics[
            [
                "Model_Label",
                "MAE",
                "RMSE",
                "R2",
                "Directional_Accuracy",
                "Train_Rows",
                "Validation_Rows",
                "Test_Rows",
            ]
        ].rename(columns={"Model_Label": "Model"}).style.format(
            {
                "MAE": "{:.4f}",
                "RMSE": "{:.4f}",
                "R2": "{:.4f}",
                "Directional_Accuracy": "{:.2f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# --------------------------------------------------
# Tab 4: Methodology
# --------------------------------------------------
with tab4:
    st.subheader("Project Methodology")

    st.markdown(
        """
        **Markets covered**

        - Indian equities: RELIANCE, TCS, HDFCBANK, ICICIBANK, INFY
        - US equities: AAPL, MSFT, JPM, XOM, NVDA

        **Prediction target**

        Next available trading-session adjusted-close return.

        **Feature engineering**

        - Daily and lagged returns
        - RSI and MACD indicators
        - Bollinger Band width
        - ATR-normalized volatility
        - Rolling volatility measures
        - Volume features
        - Calendar features

        **Models evaluated**

        - Zero-return, historical mean and median baselines
        - Previous-day return baseline
        - Linear Regression and Ridge Regression
        - Random Forest and XGBoost
        - Long Short-Term Memory (LSTM)

        **Evaluation design**

        Chronological train, validation and test splits were used.
        Boundary observations were purged to prevent next-day target leakage.
        The final test set was kept separate from model training and selection.
        """
    )

    st.divider()

    st.subheader("Key Research Findings")

    st.markdown(
        f"""
        - The best overall model by average MAE was
          **{MODEL_LABELS.get(overall_best['Model'], overall_best['Model'])}**,
          with MAE **{overall_best['Average_MAE']:.4f}**.
        - The zero-return baseline achieved an average MAE of
          **{summary.loc[summary['Model'] == 'Zero_Return', 'Average_MAE'].iloc[0]:.4f}**.
        - Several trained models did not outperform simple baselines.
        - Average R² values were negative across the evaluated model set.
        - Directional accuracy was generally close to 50%.
        """
    )

    st.warning(
        "Research conclusion: The current experiments do not establish "
        "reliable predictive or trading performance. Stronger validation, "
        "feature research, transaction-cost modelling and risk analysis "
        "would be required before any practical financial application."
    )

    st.subheader("Evaluation Dataset")

    st.write(
        f"- Total stock/model evaluations: {len(metrics)}"
        f"\n- Total historical prediction rows: {len(predictions):,}"
        f"\n- Number of stocks: {metrics['Ticker'].nunique()}"
        f"\n- Number of model candidates: {metrics['Model'].nunique()}"
    )

    st.download_button(
        "Download complete model metrics (CSV)",
        data=metrics.to_csv(index=False).encode("utf-8"),
        file_name="final_test_metrics.csv",
        mime="text/csv",
    )

st.divider()

st.caption(
    "Cross-Market Stock Return Forecasting | "
    "Machine Learning & Deep Learning Research Project"
)
