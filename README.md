# Systematic Alpha Signal Engine (SASE)

SASE is an automated, event-driven quantitative trading pipeline designed to detect pre-market momentum anomalies in US micro and small-cap equities. The engine executes daily at 07:30 EST, orchestrating a multi-stage data science workflow that fuses quantitative price action, Natural Language Processing (NLP), and Retrieval-Augmented Generation (RAG) into a single predictive trading signal.

---

## Architecture Overview

The pipeline is entirely serverless and designed to run via GitHub Actions. It utilizes a hybrid AI architecture to evaluate both the technical setup and the underlying fundamental catalyst.

### 1. Market Screening
The pipeline connects to the TradingView Scanner API to evaluate the entire US equity market (NYSE, NASDAQ, AMEX). It isolates "Stocks in Play" using strict pre-market quantitative thresholds:
- Price: $1.00 – $25.00
- Market Capitalization: $50M – $2B
- Pre-market Gap: > 3%
- Relative Volume (RVOL): Pre-market volume must exceed 5% of the 10-day Average Daily Volume (ADV).

### 2. Catalyst Extraction & NLP (FinBERT)
For each qualifying ticker, SASE fetches the latest 48 hours of news headlines via the Finnhub API.
- **Sentiment Scoring**: Headlines are scored using the `ProsusAI/finbert` sequence classification model. 
- **Time Decay**: A 3-hour half-life exponential decay function is applied to the sentiment score, penalizing stale news. (The decay clock pauses over the weekend).
- **Zero-Shot Classification**: Headlines exhibiting high absolute sentiment are routed through `gpt-4o-mini` to extract the exact catalyst category (e.g., `FDA_APPROVAL`, `EARNINGS`, `OFFERING_DILUTION`).

### 3. Historical Event Retrieval (RAG)
SASE maintains a Pinecone vector database of historical market events. 
- The extracted catalyst is embedded using OpenAI's `text-embedding-3-small`.
- A two-stage semantic search is performed. It first applies an exact-match metadata filter to pull how the *specific* ticker reacted to similar past events. If there is insufficient history, it falls back to global averages across the broader market.
- The resulting historical win-rate is passed as a continuous feature to the downstream model.

### 4. Feature Engineering
Point-in-time quantitative features are constructed using 15-minute and Daily OHLCV data from Polygon.io (adjusted for extended trading hours).
- **Core Features**: `rvol_15m`, `momentum_1h`, `rsi_14`, `sector_beta` (relative to IWM).
- **Interaction Features**: Cross-features like `vol_momentum_interaction` are engineered to capture non-linear volume/price expansion dynamics.

### 5. Inference (LightGBM)
The final feature matrix is passed to a LightGBM classifier (`models/production_alpha_model.pkl`). 
The model was trained offline using a TimeSeriesSplit cross-validation strategy to strictly prevent look-ahead bias. If the model outputs a squeeze probability > 0.85, the pipeline queues an execution alert.

### 6. Alerting
Output signals are formatted and pushed via webhook to a dedicated Discord channel for manual review or secondary automated execution.

---

## Installation & Setup

### Requirements
- Python 3.10+
- `pip install -r requirements.txt`

### Environment Variables
You must configure the following environment variables (via `.env` or GitHub Secrets) for the pipeline to authenticate with external data providers:
```env
FINNHUB_API_KEY=your_finnhub_key
MASSIVE_API_KEY=your_polygon_key
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
DISCORD_WEBHOOK_URL=your_discord_webhook
```

### Execution
To run the live production pipeline:
```bash
python main.py
```

To run a simulation (injects mock pre-market data to bypass screener constraints for testing):
```bash
python main.py --simulate
```

---

## Disclaimer
This software is provided for educational and research purposes only. The Systematic Alpha Signal Engine does not constitute financial advice. The models and logic contained herein are experimental. Users are entirely responsible for any financial losses incurred through the use or modification of this code.
