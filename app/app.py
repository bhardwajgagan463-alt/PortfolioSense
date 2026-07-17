import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import yfinance as yf
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import date, timedelta
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="PortfolioSense",
    page_icon="📈",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stAlert { border-radius: 8px; }
    h1 { color: #e2e8f0 !important; }
    .stCaption { color: #718096 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────
st.sidebar.title("⚙️ Controls")
st.sidebar.markdown("---")

# Preset popular NSE stocks
PRESET_TICKERS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS',
    'ICICIBANK.NS', 'HINDUNILVR.NS', 'BHARTIARTL.NS', 'TMPV.NS',
    'WIPRO.NS', 'AXISBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS',
    'MARUTI.NS', 'BAJFINANCE.NS', 'TITAN.NS', 'NESTLEIND.NS'
]

st.sidebar.subheader("Stock Selection")
input_mode = st.sidebar.radio(
    "How to pick stocks",
    ["Choose from list", "Type any NSE ticker"],
    help="Type any NSE ticker for stocks not in the preset list"
)

if input_mode == "Choose from list":
    selected_raw = st.sidebar.multiselect(
        "Select stocks",
        options=PRESET_TICKERS,
        default=['RELIANCE.NS', 'TCS.NS', 'INFY.NS'],
        format_func=lambda x: x.replace('.NS', '')
    )
else:
    ticker_input = st.sidebar.text_input(
        "Enter NSE tickers (comma separated)",
        value="RELIANCE.NS, TCS.NS, WIPRO.NS",
        help="Format: TICKERNAME.NS — e.g. WIPRO.NS, HCLTECH.NS"
    )
    selected_raw = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]

st.sidebar.markdown("---")
st.sidebar.subheader("Date Range")
start_date = st.sidebar.date_input(
    "Start date",
    value=date(2019, 1, 1),
    min_value=date(2010, 1, 1),
    max_value=date.today() - timedelta(days=90)
)
end_date = st.sidebar.date_input(
    "End date",
    value=date.today(),
    min_value=date(2010, 1, 1),
    max_value=date.today()
)

st.sidebar.markdown("---")
st.sidebar.subheader("Anomaly Settings")
contamination = st.sidebar.slider(
    "Anomaly sensitivity",
    min_value=0.01, max_value=0.05,
    value=0.02, step=0.005,
    help="Higher = more anomalies flagged. Default 2% (1 in 50 days)"
)

st.sidebar.markdown("---")
st.sidebar.caption("📡 Data fetched live from Yahoo Finance")
st.sidebar.caption(f"🕐 As of: {date.today().strftime('%d %b %Y')}")

# ── Title ──────────────────────────────────────────────────
st.title("📈 PortfolioSense")
st.subheader("Live Stock Risk & Anomaly Detection Dashboard")
st.markdown("---")

if not selected_raw:
    st.warning("Please select at least one stock from the sidebar to begin.")
    st.stop()

# ── Data fetching & pipeline ───────────────────────────────
@st.cache_data(ttl=3600)  # cache for 1 hour so repeated interactions are fast
def fetch_and_process(ticker, start, end, contamination_rate):
    """
    Full pipeline for a single ticker:
    fetch → clean → feature engineer → anomaly detect
    Returns processed dataframe or None on failure
    """
    try:
        raw = yf.download(ticker, start=str(start), end=str(end), progress=False)
        if raw.empty or len(raw) < 60:
            return None, f"Not enough data for {ticker}. Check ticker format (e.g. WIPRO.NS)"

        # Handle MultiIndex columns from yfinance
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = pd.DataFrame(index=raw.index)
        df['close'] = raw['Close']
        df['volume'] = raw['Volume']
        df = df.dropna()

        # Feature engineering
        df['daily_return'] = df['close'].pct_change()
        df['ma_7'] = df['close'].rolling(7).mean()
        df['ma_30'] = df['close'].rolling(30).mean()
        df['volatility_30'] = df['daily_return'].rolling(30).std()

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))

        df = df.dropna()

        # Anomaly detection
        features = ['daily_return', 'volatility_30', 'rsi_14', 'ma_7', 'ma_30']
        X = df[features].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        iso = IsolationForest(
            n_estimators=100,
            contamination=contamination_rate,
            random_state=42
        )
        iso.fit(X_scaled)
        df['anomaly_score'] = iso.decision_function(X_scaled)
        df['is_anomaly'] = (iso.predict(X_scaled) == -1).astype(int)

        return df, None

    except Exception as e:
        return None, str(e)

# ── Fetch all selected stocks ──────────────────────────────
all_data = {}
errors = []

with st.spinner("Fetching live market data and running analysis..."):
    for ticker in selected_raw:
        df, err = fetch_and_process(ticker, start_date, end_date, contamination)
        if df is not None:
            all_data[ticker] = df
        else:
            errors.append(f"**{ticker}**: {err}")

if errors:
    for e in errors:
        st.error(e)

if not all_data:
    st.error("No valid data could be fetched. Please check your ticker symbols and try again.")
    st.stop()

# Clean display names
def clean_name(ticker):
    return ticker.replace('.NS', '').replace('.BO', '')

# ── Section 1: Price Charts ────────────────────────────────
st.header("📊 Price Charts with Anomaly Detection")
st.caption("Red dots mark statistically unusual price behavior detected by Isolation Forest")

for ticker, df in all_data.items():
    name = clean_name(ticker)
    anomaly_points = df[df['is_anomaly'] == 1]
    anomaly_rate = df['is_anomaly'].mean() * 100

    col_title, col_stats = st.columns([3, 1])
    with col_title:
        st.subheader(f"{name}")
    with col_stats:
        current_price = df['close'].iloc[-1]
        total_return = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100
        st.metric("Current Price", f"₹{current_price:,.0f}",
                  delta=f"{total_return:+.1f}% since {start_date.year}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7),
                                    gridspec_kw={'height_ratios': [3, 1]},
                                    sharex=True)
    fig.patch.set_facecolor('#0e1117')
    ax1.set_facecolor('#0e1117')
    ax2.set_facecolor('#0e1117')

    # Price + MAs
    ax1.plot(df.index, df['close'], color='#4299e1', linewidth=1, label='Close Price', alpha=0.9)
    ax1.plot(df.index, df['ma_7'], color='#68d391', linewidth=0.8, alpha=0.6, label='7-day MA')
    ax1.plot(df.index, df['ma_30'], color='#f6ad55', linewidth=0.8, alpha=0.6, label='30-day MA')
    ax1.scatter(anomaly_points.index, anomaly_points['close'],
                color='#fc8181', s=45, zorder=5, label=f'Anomaly ({len(anomaly_points)} days)', alpha=0.9)
    ax1.set_ylabel('Price (INR)', color='#a0aec0')
    ax1.tick_params(colors='#a0aec0')
    ax1.legend(loc='upper left', facecolor='#1a1f2e', edgecolor='#2d3748',
               labelcolor='#e2e8f0', fontsize=9)
    for spine in ax1.spines.values():
        spine.set_color('#2d3748')

    # Anomaly score
    ax2.plot(df.index, df['anomaly_score'], color='#9f7aea', linewidth=0.8, alpha=0.8)
    ax2.axhline(y=0, color='#fc8181', linestyle='--', alpha=0.5, linewidth=0.8)
    ax2.fill_between(df.index, df['anomaly_score'], 0,
                     where=(df['anomaly_score'] < 0),
                     color='#fc8181', alpha=0.2)
    ax2.set_ylabel('Anomaly\nScore', color='#a0aec0', fontsize=8)
    ax2.tick_params(colors='#a0aec0', labelsize=8)
    for spine in ax2.spines.values():
        spine.set_color('#2d3748')

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax2.xaxis.get_majorticklabels(), color='#a0aec0')

    plt.tight_layout(pad=1.5)
    st.pyplot(fig)
    plt.close()

    # Top anomaly dates
    if len(anomaly_points) > 0:
        top_anomalies = anomaly_points['daily_return'].abs().nlargest(3)
        dates_str = ', '.join([f"{d.strftime('%Y-%m-%d')} ({anomaly_points.loc[d, 'daily_return']*100:+.1f}%)"
                               for d in top_anomalies.index])
        st.caption(f"🔴 Most extreme anomaly days: {dates_str}")

    st.markdown("---")

# ── Section 2: Portfolio Risk ──────────────────────────────
st.header("⚠️ Portfolio Risk Summary")

returns_df = pd.DataFrame({
    clean_name(t): df['daily_return']
    for t, df in all_data.items()
})

weights = np.array([1/len(all_data)] * len(all_data))
cov_matrix = returns_df.cov() * 252
port_vol = np.sqrt(weights.T @ cov_matrix.values @ weights)
port_daily = returns_df.mean(axis=1)
var_95 = np.percentile(port_daily, 5)
var_99 = np.percentile(port_daily, 1)
individual_vol = returns_df.std() * np.sqrt(252)
weighted_avg_vol = np.sum(weights * individual_vol.values)
diversification_benefit = (weighted_avg_vol - port_vol) * 100

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Portfolio Volatility", f"{port_vol*100:.2f}%",
            help="Annualized combined portfolio volatility")
col2.metric("VaR 95%", f"{var_95*100:.2f}%",
            help="Max expected daily loss on 95% of trading days")
col3.metric("VaR 99%", f"{var_99*100:.2f}%",
            help="Max expected daily loss on 99% of trading days")
col4.metric("Diversification Benefit", f"{diversification_benefit:.2f}%",
            help="Volatility reduction from holding multiple uncorrelated stocks")
col5.metric("Stocks Analyzed", len(all_data))

st.markdown("---")

# Correlation heatmap + Risk ranking side by side
col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.subheader("Correlation Matrix")
    if len(all_data) > 1:
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        sns.heatmap(returns_df.corr(), annot=True, cmap='RdYlGn_r',
                    center=0, fmt='.2f', ax=ax,
                    annot_kws={'size': 9, 'color': 'white'},
                    linewidths=0.5, linecolor='#2d3748')
        ax.set_title('Daily Returns Correlation', color='#e2e8f0', pad=10)
        ax.tick_params(colors='#a0aec0', labelsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("Values closer to 1.0 = stocks move together = less diversification benefit")
    else:
        st.info("Select more than one stock to see correlation matrix.")

with col_right:
    st.subheader("Risk Ranking")
    risk_df = pd.DataFrame({
        'Ann. Volatility': (returns_df.std() * np.sqrt(252)).round(4),
        'Mean Daily Return': returns_df.mean().round(6),
        'Sharpe Proxy': ((returns_df.mean() * 252) /
                         (returns_df.std() * np.sqrt(252))).round(3),
        'Anomaly Days': [all_data[t]['is_anomaly'].sum()
                         for t in all_data.keys()]
    }).sort_values('Ann. Volatility', ascending=False)
    st.dataframe(risk_df, use_container_width=True, height=300)
    st.caption("Sharpe Proxy = return per unit of risk. Higher = more efficient.")

st.markdown("---")

# ── Section 3: Anomaly Explorer ───────────────────────────
st.header("🔍 Anomaly Explorer")

selected_stock = st.selectbox(
    "Select a stock to explore",
    options=list(all_data.keys()),
    format_func=clean_name
)

df_sel = all_data[selected_stock]
anomaly_days = df_sel[df_sel['is_anomaly'] == 1].copy()
anomaly_days['return_pct'] = (anomaly_days['daily_return'] * 100).round(2)

st.markdown(f"**{len(anomaly_days)} anomalous days detected for {clean_name(selected_stock)}** "
            f"({anomaly_days['daily_return'].mean()*100:+.2f}% avg return on anomaly days)")

# Extreme anomaly table
st.subheader("Most Extreme Anomaly Days")
extreme = anomaly_days[['close', 'return_pct', 'volatility_30', 'rsi_14', 'anomaly_score']] \
    .sort_values('return_pct', key=abs, ascending=False) \
    .head(10) \
    .round(4)
extreme.index = extreme.index.strftime('%Y-%m-%d')
extreme.columns = ['Close Price', 'Return %', '30d Volatility', 'RSI (14)', 'Anomaly Score']
st.dataframe(extreme, use_container_width=True)

# Anomaly score chart
st.subheader("Anomaly Score Over Time")
fig, ax = plt.subplots(figsize=(14, 3.5))
fig.patch.set_facecolor('#0e1117')
ax.set_facecolor('#0e1117')
ax.plot(df_sel.index, df_sel['anomaly_score'],
        color='#9f7aea', linewidth=0.8, alpha=0.9)
ax.axhline(y=0, color='#fc8181', linestyle='--', alpha=0.6, linewidth=1)
ax.fill_between(df_sel.index, df_sel['anomaly_score'], 0,
                where=(df_sel['anomaly_score'] < 0),
                color='#fc8181', alpha=0.25, label='Anomalous region')
ax.set_ylabel('Anomaly Score', color='#a0aec0')
ax.tick_params(colors='#a0aec0')
for spine in ax.spines.values():
    spine.set_color('#2d3748')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.setp(ax.xaxis.get_majorticklabels(), color='#a0aec0')
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.caption("💡 Scores below 0 = anomalous behavior. More negative = more extreme. "
           "Adjust sensitivity in the sidebar to flag more or fewer days.")

st.markdown("---")
st.caption("⚠️ PortfolioSense is for educational and analytical purposes only. "
           "Not financial advice. Data sourced from Yahoo Finance via yfinance.")