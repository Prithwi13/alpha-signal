<div align="center">
  <h1>📈 Systematic Alpha Signal Engine (SASE)</h1>
  <p>An institutional-grade, fully automated quantitative trading pipeline that screens the US equity market, analyzes breaking news via NLP, builds point-in-time quantitative features, and generates highly precise predictive trading signals using Machine Learning.</p>
</div>

---

## 🌟 Overview

The **Systematic Alpha Signal Engine (SASE)** is a robust, cloud-native automated quant pipeline designed to identify micro-cap and small-cap momentum squeezes in the pre-market. Every morning at 7:30 AM EST, SASE automatically wakes up via GitHub Actions, executes its 5-stage data science pipeline, and pushes high-probability trading alerts directly to a Discord webhook.

Unlike typical momentum screeners, SASE utilizes a hybrid AI architecture:
- **FinBERT NLP** for real-time sentiment extraction.
- **OpenAI + LangChain** for precise Catalyst categorization (FDA Approvals, Earnings, Contracts).
- **Pinecone Vector Database (RAG)** to query historical win-rates of similar past catalysts.
- **LightGBM / XGBoost** to synthesize the quantitative features into a final probabilistic trade signal.

## ⚙️ Architecture Pipeline

The pipeline is orchestrated in `main.py` and runs completely autonomously.

1. **The Screener (`src/screener.py`)**
   - Connects to the free TradingView Scanner API to bypass typical data provider rate limits.
   - Filters the entire US market for "Stocks in Play" (Price $1-$25, Market Cap $50M-$2B, Gap > 3%, RVOL > 2.5).

2. **The NLP Engine (`src/nlp_engine.py`)**
   - Fetches the latest 48-hour news headlines via Finnhub.
   - Evaluates sentiment using HuggingFace's **ProsusAI/finbert**, applying a time-decay mathematical formula to prioritize breaking news.
   - Uses **GPT-4o-mini** via LangChain to extract the precise catalyst (e.g., `FDA_APPROVAL`, `EARNINGS`).

3. **The RAG Store (`src/rag_store.py`)**
   - Embeds the catalyst and queries a **Pinecone** vector database using OpenAI's `text-embedding-3-small`.
   - Looks for similar historical events and returns a mathematical historical win-rate for that specific setup.

4. **The Feature Builder (`src/feature_builder.py`)**
   - Uses the **Massive API (Polygon.io)** to fetch real-time 15m and Daily candlesticks (with strict free-tier rate limit guards).
   - Engineers point-in-time quantitative features: `rvol_15m`, `momentum_1h`, `rsi_14`, `sector_beta`, and `excess_momentum`.
   - Incorporates engineered interaction features (e.g. `vol_momentum_interaction`) to capture non-linear relationships.

5. **Model Inference (`src/model_inference.py`)**
   - Passes the final feature matrix into a heavily regularized, TimeSeries-validated **LightGBM** machine learning model (`models/production_alpha_model.pkl`).
   - If the model predicts a squeeze with a probability > 0.85, it queues an alert.

6. **Notification System (`src/notifier.py`)**
   - Formats the resulting alpha predictions into a rich, color-coded embed.
   - Sends the report to a Discord channel via Webhook.

## 🔬 Data Science & Validation

The codebase includes a strict, institutional-grade quant pipeline (`scripts/quant_pipeline.py`) used to train the production model:
- **No Look-Ahead Bias**: Model validation strictly utilizes `TimeSeriesSplit(n_splits=5)`.
- **Statistically Significant Features**: Core features like RVOL have been verified via T-Tests (p < 0.05).
- **Overfitting Prevention**: The ML models are constrained (`max_depth=3`, `subsample=0.8`) to ensure robust out-of-sample performance on highly noisy financial data.

## 🚀 Deployment (GitHub Actions)

SASE is built to run flawlessly in the cloud without requiring a dedicated server.

- **Workflow**: `.github/workflows/daily_pipeline.yml`
- **Schedule**: Cron triggered every Monday-Friday at `11:30 UTC` (7:30 AM EST, exactly 2 hours before market open).
- **Environment**: Installs requirements, provisions the NLP models, executes `main.py`, and spins down securely.

### Environment Variables Required
To run this project, the following secrets must be set in your GitHub repository or local `.env` file:
- `FINNHUB_API_KEY`
- `MASSIVE_API_KEY` (Polygon.io)
- `OPENAI_API_KEY`
- `PINECONE_API_KEY`
- `DISCORD_WEBHOOK_URL`

## 🛠️ Usage

### Run Live Pipeline (Local)
```bash
python main.py
```

### Run Simulation (Bypasses Screener to test Webhooks & Inference)
```bash
python main.py --simulate
```

### Retrain Production Model (Quant Pipeline)
```bash
python scripts/quant_pipeline.py
```
