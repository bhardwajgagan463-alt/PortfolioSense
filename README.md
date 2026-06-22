# 📈 PortfolioSense
### Stock Risk & Anomaly Detection Dashboard

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.51.0-red?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![Prophet](https://img.shields.io/badge/Prophet-Forecasting-green)

> An end-to-end data science project that monitors a portfolio of Indian NSE-listed stocks, forecasts future price trends, and automatically detects unusual market behavior — presented through an interactive web dashboard.

🔗 **[Live Demo → portfoliosense.streamlit.app](#)** *(replace with your deployed URL)*

---

## 📌 What It Does

Individual investors and small portfolio managers in India face a common problem: they hold a basket of stocks but have no systematic way to know when a stock is behaving unusually, where prices might be heading, or how risky their overall portfolio really is.

**PortfolioSense solves this by combining three things in one dashboard:**

- 🔮 **Price Forecasting** — projects where each stock is headed over the next 90 days using a time-series model, with a confidence band showing uncertainty
- 🚨 **Anomaly Detection** — automatically flags unusual market behavior (sudden price crashes, abnormal volume spikes) using an unsupervised ML model trained on 7+ years of historical data
- 📊 **Portfolio Risk Analysis** — quantifies overall portfolio risk, diversification benefit, Value-at-Risk, and identifies which stocks contribute most to risk

---

## ✨ Key Features

- Interactive stock picker — select any combination of 8 NSE stocks to analyze
- Price chart with anomaly flags and 90-day forecast overlaid in one view
- Portfolio volatility and Value-at-Risk (VaR) calculated dynamically for selected stocks
- Correlation heatmap showing sector clustering and hidden concentration risk
- Risk ranking table with annualized volatility and Sharpe proxy per stock
- Anomaly Explorer — drill into any stock's most extreme days with full feature breakdown
- Anomaly score chart with anomalous periods visually highlighted

---

## 📊 Key Findings from Analysis

These are real findings from 7+ years (2019–2026) of NSE stock data — not placeholder results:

| Finding | Detail |
|---|---|
| **COVID crash detected automatically** | Isolation Forest flagged March 2020 as the most anomalous period across all 8 stocks independently, with no labels or prior knowledge |
| **TMPV demerger correctly identified** | The Oct 14 2025 Tata Motors -40% day was flagged as a stock-specific structural event (demerger), correctly distinguished from a market crash |
| **TCS June 2026 selloff caught** | -8.39% single-day drop driven by AI/macro fears flagged as anomalous — validated against real news |
| **Diversification benefit: 10.48%** | Full 8-stock portfolio achieves 18.33% annualized volatility vs 28.82% with no diversification — a 10.48% reduction purely from portfolio construction |
| **Sector concentration risk identified** | HDFC Bank / ICICI Bank show 0.62 correlation; TCS / Infosys show 0.67 — both pairs reduce effective diversification |
| **Best forecasting accuracy** | Infosys: MAPE 2.94%, TCS: 3.31%, HUL: 4.47% — stable low-volatility stocks are most predictable |
| **Hardest to forecast** | HDFC Bank: MAPE 28.50% — driven by the 2023 HDFC merger disrupting historical price patterns |

---

## 🧠 ML / AI Methodology

### 1. Anomaly Detection — Isolation Forest (Unsupervised)
Isolation Forest is trained per stock on 5 features: daily return, 30-day rolling volatility, 14-day RSI, 7-day MA, and 30-day MA. It learns what "normal" behavior looks like by randomly partitioning the feature space and identifying data points that are isolated quickly. No manual labeling required.

- **Contamination rate:** 2% (approx. 37 anomalous days per stock over 7 years)
- **Validation:** Correctly flagged March 2020 COVID crash, Oct 2025 TMPV demerger, and June 2026 TCS selloff — three completely different anomaly types

### 2. Price Forecasting — Facebook Prophet
A separate Prophet model is trained per stock on historical closing prices. Prophet decomposes the time series into trend + seasonality + noise, then extends the trend forward 90 business days with upper/lower confidence bounds.

| Stock | RMSE | MAPE |
|---|---|---|
| INFY | 44.52 | 2.94% |
| TCS | 95.62 | 3.31% |
| HINDUNILVR | 109.93 | 4.47% |
| RELIANCE | 134.87 | 8.84% |
| ICICIBANK | 161.14 | 12.20% |
| BHARTIARTL | 315.29 | 16.99% |
| TMPV | 91.59 | 21.43% |
| HDFCBANK | 220.92 | 28.50% |

> ⚠️ **Honest disclaimer:** Stock price forecasting has inherent uncertainty due to market efficiency. The value of this component is demonstrating time-series methodology and honest evaluation — not claiming the ability to predict markets reliably.

### 3. Portfolio Risk — Statistical Methods
- **Annualized Volatility:** Daily return std × √252
- **Portfolio Volatility:** Computed using covariance matrix and equal weights
- **Value-at-Risk (VaR):** Historical simulation at 95% and 99% confidence
- **Sharpe Proxy:** (Mean annual return) / (Annualized volatility)

---

## 🛠️ Tech Stack

| Category | Technology | Purpose |
|---|---|---|
| Data Collection | yfinance | Pull live/historical NSE OHLCV data |
| Data Manipulation | Pandas, NumPy | Cleaning, feature engineering, date handling |
| Visualization | Matplotlib, Seaborn | EDA charts, heatmaps, rolling plots |
| Machine Learning | Scikit-learn | Isolation Forest anomaly detection |
| Forecasting | Prophet | Time-series price trend forecasting |
| Frontend / Dashboard | Streamlit | Interactive web dashboard |
| Deployment | Streamlit Community Cloud | Free hosting, public shareable URL |
| Version Control | Git / GitHub | Code management and project showcase |

---

## 📁 Project Structure

```
PortfolioSense/
│
├── notebooks/
│   ├── 01_data_collection.ipynb       # Pull and save raw stock data
│   ├── 02_eda.ipynb                   # Exploratory data analysis
│   ├── 03_feature_engineering.ipynb   # Build feature set per stock
│   ├── 04_forecasting.ipynb           # Train Prophet models, evaluate
│   ├── 05_anomaly_detection.ipynb     # Train Isolation Forest, validate
│   └── 06_portfolio_risk.ipynb        # Portfolio-level risk metrics
│
├── app/
│   └── app.py                         # Streamlit dashboard
│
├── data/
│   ├── raw_stock_data.csv             # Raw OHLCV data for all 8 stocks
│   ├── features/                      # Feature-engineered CSV per stock
│   ├── forecasts/                     # Prophet forecast outputs per stock
│   ├── anomalies/                     # Anomaly detection results per stock
│   └── portfolio/                     # Portfolio-level risk metrics
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/PortfolioSense.git
cd PortfolioSense
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn prophet yfinance
```

### 3. Run the notebooks in order
Open Jupyter Notebook (via Anaconda) and run each notebook in sequence:
```
notebooks/01_data_collection.ipynb   ← generates data/raw_stock_data.csv
notebooks/02_eda.ipynb
notebooks/03_feature_engineering.ipynb  ← generates data/features/
notebooks/04_forecasting.ipynb       ← generates data/forecasts/
notebooks/05_anomaly_detection.ipynb ← generates data/anomalies/
notebooks/06_portfolio_risk.ipynb    ← generates data/portfolio/
```

### 4. Launch the dashboard
```bash
cd app
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## ➕ How to Add a New Stock

PortfolioSense is built to be extensible. To add a new NSE-listed stock (e.g., Wipro):

### Step 1: Add the ticker to data collection
Open `01_data_collection.ipynb` and add `"WIPRO.NS"` to the tickers list:
```python
tickers = ["RELIANCE.NS", "TCS.NS", ..., "WIPRO.NS"]
```
Re-run the notebook to pull and save updated data.

### Step 2: Build features for the new stock
Open `03_feature_engineering.ipynb` — since it loops over all tickers automatically, just re-run it. It will generate `data/features/WIPRO_features.csv` automatically.

### Step 3: Train forecasting model
Open `04_forecasting.ipynb` and re-run — the loop will pick up the new ticker and generate `data/forecasts/WIPRO_forecast.csv`.

### Step 4: Run anomaly detection
Open `05_anomaly_detection.ipynb` and re-run — generates `data/anomalies/WIPRO_anomalies.csv`.

### Step 5: Update the dashboard
Open `app/app.py` and add the new ticker to the tickers list:
```python
tickers = ['BHARTIARTL', 'HDFCBANK', ..., 'WIPRO']
```

### Step 6: Re-run portfolio risk notebook
Open `06_portfolio_risk.ipynb` and re-run to recalculate portfolio metrics with the new stock included.

> 💡 **Tip:** Any NSE-listed stock works — just use the Yahoo Finance ticker format: `STOCKNAME.NS`. You can verify tickers at [finance.yahoo.com](https://finance.yahoo.com).

---

## 🔮 Future Improvements

- **Live data feed** — replace static CSV loading with real-time yfinance API calls so the dashboard always shows current data
- **LSTM comparison** — benchmark Prophet against a deep learning model (LSTM) for forecasting accuracy
- **News sentiment integration** — scrape financial news headlines and correlate sentiment scores with detected anomalies
- **Email/SMS alerts** — notify users when a new anomaly is detected in their portfolio
- **Portfolio optimization** — add a mean-variance optimization layer to suggest optimal stock weightings based on risk tolerance
- **More Indian stocks** — expand beyond 8 stocks to full Nifty 50 coverage

---

## 👤 About

Built independently as part of a Summer Training Certificate Program (2026).
Graduating 2028 | Interested in Data Science & Finance
