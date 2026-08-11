<div align="center">
  <h1>📈 Systematic Alpha Signal Engine (SASE)</h1>
  <p>An institutional-grade, fully automated quantitative trading pipeline specializing in pre-market momentum anomalies and catalyst-driven volatility in US micro and small-cap equities.</p>

  <a href="https://discord.gg/TSnPyPZGj"><img src="https://img.shields.io/badge/Discord-Join%20Live%20Signals-7289da?style=for-the-badge&logo=discord&logoColor=white" alt="Discord Server"></a>
</div>

---

## 🏛️ Strategy Overview

The **Systematic Alpha Signal Engine (SASE)** is a cloud-native, serverless quantitative trading engine. It exploits microstructural inefficiencies in the pre-market session (04:00 - 09:30 EST) where retail sentiment and institutional algorithmic flows disproportionately impact illiquid small-cap equities. 

Unlike traditional screener bots that rely purely on lagging news or static price gaps, SASE utilizes a sophisticated hybrid architecture. It synthesizes point-in-time quantitative price action, Natural Language Processing (NLP), and Retrieval-Augmented Generation (RAG) into a highly regularized Machine Learning classifier (LightGBM) to predict the probability of a sustained intraday squeeze vs. a pre-market fade.

**Live Signals & Community**: Join the automated Discord server here: [SASE Discord Server (ID: 1536523086801936454)](https://discord.gg/TSnPyPZGj)

---

## ⚙️ Data Flow & Architecture

The pipeline executes autonomously at 07:30 EST via GitHub Actions, processing data through five distinct micro-services.

### 1. Pre-Market Filtering & Screener
The pipeline connects to the TradingView Scanner API, sweeping the entire US equity market (NYSE, NASDAQ, AMEX). To qualify as "In Play", a ticker must satisfy strict statistical thresholds:
- **Price**: $1.00 – $25.00
- **Market Capitalization**: $50M – $2B
- **Pre-market Gap**: > 3%
- **Relative Volume (RVOL)**: Pre-market volume strictly verified to exceed 5% of the 10-day Average Daily Volume (ADV).

Once the entire market is screened, SASE systematically isolates only the top 15 candidates exhibiting a true institutional footprint. We deliberately narrow our focus to these specific "Stocks in Play" before executing any computationally expensive NLP or RAG logic. This funnel approach ensures the pipeline only evaluates equities that already have mathematical proof of liquidity and genuine market interest.

### 2. FinBERT NLP & Time-Decayed Sentiment
For qualifying tickers, SASE ingests the latest 48 hours of news headlines via the Finnhub API.
- **Sentiment Scoring**: Headlines are passed through HuggingFace's `ProsusAI/finbert` sequence classification model, outputting raw probabilistic sentiment vectors.
- **Weekend-Bridged Time Decay**: Markets price in information rapidly. To mimic this, SASE applies an exponential decay function with a 3-hour half-life ($S_{decayed} = S \times e^{-\lambda t}$). The mathematical clock automatically pauses over weekends (subtracting 48 hours) to ensure Friday after-hours catalysts retain their momentum going into Monday's pre-market.
- **Zero-Shot Catalyst Extraction**: Highly impactful headlines are routed through `gpt-4o-mini` to classify the exact catalyst taxonomy (e.g., `FDA_APPROVAL`, `EARNINGS`, `OFFERING_DILUTION`).

### 3. RAG-Powered Historical Analytics (Pinecone)
SASE maintains a Pinecone vector database of historical market events. 
- The extracted catalyst is embedded into a 1536-dimensional vector using OpenAI's `text-embedding-3-small`.
- **Two-Stage Exact-Match Search**: The RAG first applies an exact-match Metadata Filter to pull the *specific ticker's* historical reaction to similar past events. If there is insufficient data (common in micro-caps), it executes a secondary global semantic search to pull broader market reactions to that catalyst. The aggregated historical win-rate is fed directly to the ML model.

### 4. Feature Engineering & Quant Analysis
Point-in-time features are constructed using 15-minute OHLCV data from Polygon.io. A known train-serve skew between RTH (Regular Trading Hours) and ETH (Extended Trading Hours) is mathematically mitigated by utilizing an Exponential Moving Average (`ewm(span=100)`) for volume profiling.

**Engineered Alpha Signals:**
- `rvol_15m`: 15-minute Relative Volume (Statistically validated via T-Tests on winners vs. losers with $p < 0.05$).
- `excess_momentum`: The ticker's momentum linearly orthogonalized against the Russell 2000 (`IWM`) using a 30-day rolling `sector_beta`.
- `vol_momentum_interaction`: A non-linear cross-feature (`rvol_15m` $\times$ `momentum_1h`) allowing the tree-based model to capture explosive volume/price expansion pairs.

### 5. Machine Learning & Risk Mitigation
The final feature matrix is evaluated by a **LightGBM Classifier** (`models/production_alpha_model.pkl`). 

**Overfitting Prevention & Risk Management:**
Financial data is notoriously noisy. To ensure the model detects true signal rather than curve-fitting the noise:
1. **TimeSeries Cross-Validation**: The model was trained using `TimeSeriesSplit(n_splits=5)`. Training folds strictly precede validation folds chronologically, resulting in zero look-ahead bias and true out-of-sample ROC-AUC metrics.
2. **Heavy Regularization**: The LightGBM and XGBoost models are constrained with `max_depth=3` and `subsample=0.8` to prevent leaf memorization.
3. **News Inaccuracy & Trap Avoidance**: News is inherently lagging, frequently inaccurate, and highly susceptible to manipulation by insiders attempting to create "exit liquidity" events (where retail buys a positive headline, but institutions secretly dump their shares). SASE treats news as a lagging, untrustworthy indicator. By strictly weighting the NLP sentiment against actual Quantitative Price Action, the model mathematically mitigates this risk. If FinBERT detects a highly positive "breakthrough" headline but `excess_momentum` is heavily negative and RVOL is spiking, the model recognizes the institutional dumping trap and immediately assigns a 0% squeeze probability.

---

## 🚀 Installation & Deployment

### Requirements
- Python 3.10+
- `pip install -r requirements.txt`

### Environment Configuration
Configure the following secrets (via `.env` or GitHub Secrets) for secure API authentication:
```env
FINNHUB_API_KEY=your_finnhub_key
MASSIVE_API_KEY=your_polygon_key
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
DISCORD_WEBHOOK_URL=your_discord_webhook
```

### Execution Commands
Run the live production pipeline:
```bash
python main.py
```

Run a local quantitative simulation (injects deterministic mock data to bypass live screener constraints):
```bash
python main.py --simulate
```

Retrain the production model locally with updated historical data:
```bash
python scripts/quant_pipeline.py
```

---

## ⚖️ Disclaimer
This repository and its codebase are strictly for educational and quantitative research purposes. The Systematic Alpha Signal Engine does not constitute financial advice or an endorsement to trade. Algorithmic trading in micro-cap equities carries a high risk of total loss. Users are entirely responsible for their own risk management and financial decisions.
