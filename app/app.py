import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="PortfolioSense",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("📈 PortfolioSense")
st.subheader("Stock Risk & Anomaly Detection Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.title("Controls")
st.sidebar.markdown("---")
tickers = ['BHARTIARTL', 'HDFCBANK', 'HINDUNILVR', 'ICICIBANK',
           'INFY', 'RELIANCE', 'TCS', 'TMPV']

selected_tickers = st.sidebar.multiselect(
    "Select Stocks",
    options=tickers,
    default=['RELIANCE', 'TCS', 'INFY']
)

st.sidebar.markdown("---")
st.sidebar.caption("📅 Data last updated: June 2026")
st.sidebar.caption("Forecasts are static — regenerate notebooks for latest data.")

# ── Resolve base data path (works locally AND on Streamlit Cloud) ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')

# ── Data Loading ───────────────────────────────────────────
@st.cache_data
def load_data(ticker):
    try:
        features = pd.read_csv(
            os.path.join(DATA_DIR, 'features', f'{ticker}_features.csv'),
            index_col=0, parse_dates=True)
        anomalies = pd.read_csv(
            os.path.join(DATA_DIR, 'anomalies', f'{ticker}_anomalies.csv'),
            index_col=0, parse_dates=True)
        forecast = pd.read_csv(
            os.path.join(DATA_DIR, 'forecasts', f'{ticker}_forecast.csv'),
            index_col=0, parse_dates=True)
        return features, anomalies, forecast
    except FileNotFoundError:
        st.error(f"Data files for {ticker} not found. "
                 f"Please run notebooks 03-05 for this ticker first.")
        return None, None, None

# ── Section 1: Price Chart with Anomalies & Forecast ──────
st.header("📊 Price Chart")

if not selected_tickers:
    st.warning("Please select at least one stock from the sidebar.")
else:
    for ticker in selected_tickers:
        features, anomalies, forecast = load_data(ticker)
        if features is None:
            continue

        fig, ax = plt.subplots(figsize=(14, 5))

        # Actual price
        ax.plot(features.index, features['close'],
                label='Close Price', color='steelblue', linewidth=1)

        # Anomaly points
        anomaly_points = anomalies[anomalies['is_anomaly'] == 1]
        ax.scatter(anomaly_points.index, anomaly_points['close'],
                   color='red', s=40, zorder=5, label='Anomaly', alpha=0.8)

        # Forecast
        forecast.index = pd.to_datetime(forecast['ds'])
        future_forecast = forecast[forecast.index > features.index.max()]
        ax.plot(future_forecast.index, future_forecast['yhat'],
                color='orange', linewidth=1.5, label='Forecast')
        ax.fill_between(future_forecast.index,
                        future_forecast['yhat_lower'],
                        future_forecast['yhat_upper'],
                        alpha=0.3, color='orange')

        ax.set_title(f'{ticker} — Price, Anomalies & Forecast')
        ax.set_ylabel('Price (INR)')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.caption("⚠️ Forecasts are statistical projections, not financial advice. "
                   "Actual prices may differ significantly.")

# ── Section 2: Portfolio Risk Summary ─────────────────────
st.markdown("---")
st.header("⚠️ Portfolio Risk Summary")

if selected_tickers:
    returns_dict = {}
    for ticker in selected_tickers:
        features, _, _ = load_data(ticker)
        if features is not None:
            returns_dict[ticker] = features['daily_return']

    if returns_dict:
        returns_df = pd.DataFrame(returns_dict)

        weights = np.array([1/len(returns_dict)] * len(returns_dict))
        cov_matrix = returns_df.cov() * 252
        port_vol = np.sqrt(weights.T @ cov_matrix.values @ weights)
        port_daily_returns = returns_df.mean(axis=1)
        var_95 = np.percentile(port_daily_returns, 5)
        var_99 = np.percentile(port_daily_returns, 1)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Portfolio Volatility", f"{port_vol*100:.2f}%",
                    help="Annualized portfolio volatility")
        col2.metric("VaR 95%", f"{var_95*100:.2f}%",
                    help="Max daily loss on 95% of days")
        col3.metric("VaR 99%", f"{var_99*100:.2f}%",
                    help="Max daily loss on 99% of days")
        col4.metric("Stocks Selected", len(returns_dict))

        st.subheader("Correlation Matrix")
        if len(returns_dict) > 1:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(returns_df.corr(), annot=True, cmap='coolwarm',
                        center=0, fmt='.2f', ax=ax)
            ax.set_title('Daily Returns Correlation')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Select more than one stock to see correlation matrix.")

        st.subheader("Risk Ranking")
        risk_df = pd.DataFrame({
            'Annualized Volatility': (returns_df.std() * np.sqrt(252)).round(4),
            'Mean Daily Return': returns_df.mean().round(6),
            'Sharpe Proxy': ((returns_df.mean() * 252) /
                             (returns_df.std() * np.sqrt(252))).round(4)
        }).sort_values('Annualized Volatility', ascending=False)
        st.dataframe(risk_df, use_container_width=True)

# ── Section 3: Anomaly Explorer ───────────────────────────
st.markdown("---")
st.header("🔍 Anomaly Explorer")

if selected_tickers:
    selected_stock = st.selectbox(
        "Select a stock to explore anomalies",
        options=selected_tickers
    )

    _, anomalies, _ = load_data(selected_stock)
    if anomalies is not None:
        anomaly_days = anomalies[anomalies['is_anomaly'] == 1].copy()
        anomaly_days['return_pct'] = (anomaly_days['daily_return'] * 100).round(2)

        st.markdown(f"**{len(anomaly_days)} anomalous days detected for {selected_stock}**")

        # Known events context
        known_events = {
            'TMPV': '⚠️ Note: The Oct 14 2025 anomaly is a Tata Motors demerger event, not a market crash.',
            'TCS': '⚠️ Note: The Jun 3 2026 anomaly reflects an AI/macro sector selloff (-8.39%).',
        }
        if selected_stock in known_events:
            st.info(known_events[selected_stock])

        st.subheader("Most Extreme Anomaly Days")
        extreme = anomaly_days[['close', 'return_pct', 'volatility_30', 'rsi_14', 'anomaly_score']]\
            .sort_values('return_pct', key=abs, ascending=False)\
            .head(10)\
            .round(4)
        extreme.index = extreme.index.strftime('%Y-%m-%d')
        st.dataframe(extreme, use_container_width=True)

        st.subheader("Anomaly Score Over Time")
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(anomalies.index, anomalies['anomaly_score'],
                color='purple', linewidth=0.8, alpha=0.8)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.6, label='Anomaly Threshold')
        ax.fill_between(anomalies.index, anomalies['anomaly_score'], 0,
                        where=(anomalies['anomaly_score'] < 0),
                        color='red', alpha=0.3, label='Anomalous Region')
        ax.set_title(f'{selected_stock} — Anomaly Score Over Time')
        ax.set_ylabel('Anomaly Score')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.caption("💡 Scores below 0 indicate anomalous behavior. "
                   "The more negative the score, the more extreme the anomaly.")