# Systematic Alpha Signal Engine (SASE)
## Technical Architecture & Quantitative Review

**Author:** Antigravity Quant Team  
**Date:** August 2026  

---

### 1. Executive Summary & USP (Unique Selling Proposition)
The Systematic Alpha Signal Engine (SASE) is an end-to-end, fully automated quantitative trading pipeline designed to exploit **pre-market micro-cap momentum inefficiencies**. 

**The USP of this project is its fusion of 3 distinct domains:**
1. **Traditional Quant Finance:** Statistical momentum, rolling beta, RVOL, and XGBoost machine learning.
2. **State-of-the-art NLP:** HuggingFace FinBERT for sentiment decay and Langchain (GPT-4o) for discrete event categorization.
3. **Retrieval-Augmented Generation (RAG):** Pinecone vector databases to fetch historical win-rates of similar past catalysts, injecting memory into the model.

By combining real-time fundamental event detection (FDA approvals, earnings) with price-action momentum, SASE avoids the common pitfall of purely technical models (whipsawing on fake volume) and purely fundamental models (buying great news that the market ignores).

---

### 2. Step-by-Step Code Analysis

#### Step 1: The Screener (`src/screener.py`)
- **What it does:** Scans the entire US equity market (NYSE, AMEX, NASDAQ) at 7:30 AM EST.
- **Quant Logic:** 
  - `abs(gap_pct) > 0.03`: Ensures the stock has moved at least 3% in the pre-market.
  - `1.00 <= price <= 25.00` and `50M <= market_cap <= 2B`: Targets the sweet spot of micro/small caps where retail euphoria and short-squeezes occur.
  - `rvol >= 2.5`: The most critical quant metric. It ensures current pre-market volume is at least 250% of the historical average, proving institutional or mass-retail involvement.
- **Advantage:** By using the unauthenticated `scanner.tradingview.com/america/scan` POST endpoint, we bypass the aggressive IP-blocking of Yahoo Finance, making the pipeline perfectly stable on cloud runners like GitHub Actions.

#### Step 2: NLP & Catalyst Engine (`src/nlp_engine.py`)
- **What it does:** Fetches the last 48 hours of news via Finnhub.
- **Quant Logic:**
  - **FinBERT Scoring:** Uses `ProsusAI/finbert` to convert raw text into a probability matrix `[positive, negative, neutral]`.
  - **Exponential Time Decay:** `S_decayed = S * exp(- (ln(2)/3.0) * t_hours)`. News that is 3 hours old is worth half as much as news that broke 5 minutes ago. This perfectly models the transient nature of alpha in small-cap equities.
  - **Langchain Extractor:** If the decayed sentiment is highly impactful (`abs(S_decayed) > 0.4`), it prompts `gpt-4o-mini` to categorize the event (e.g., `FDA_APPROVAL`, `EARNINGS`, `OFFERING_DILUTION`).

#### Step 3: Vector Memory / RAG (`src/rag_store.py`)
- **What it does:** Uses OpenAI embeddings (`text-embedding-3-small`) to embed the catalyst and queries a Pinecone Serverless database.
- **Quant Logic:** Finds the top 5 most similar historical events (e.g., "XYZ also had a Phase 2 FDA approval last month") and calculates their average historical win-rate. This effectively gives the XGBoost model "memory" of how the market historically reacts to specific narrative themes.

#### Step 4: Feature Builder (`src/feature_builder.py`)
- **What it does:** Merges the structural price data with the NLP/RAG data.
- **Quant Logic:** 
  - Calculates 15-minute `RSI` and 1-hour `momentum`.
  - Calculates 30-day rolling `sector_beta` against the IWM (Russell 2000).
  - Calculates `excess_momentum = momentum_1h - (beta_30d * iwm_momentum_1h)`. This isolates the stock's idiosyncratic alpha from broader market movements.
  - **Lookahead Guard:** Implements strict index filtering to ensure no future data leaks into the training matrix.

#### Step 5: Machine Learning (`src/model_trainer.py` & `model_inference.py`)
- **What it does:** Trains an XGBoost classifier using `TimeSeriesSplit` to prevent forward-looking bias.
- **Quant Logic:** 
  - Predicts the probability of a successful intraday squeeze.
  - Uses `scale_pos_weight=5` to handle the imbalanced nature of breakouts (most gappers fade).
  - Evaluates via Precision, Recall, and ROC-AUC. 

#### Step 6: Orchestration & Alerting (`main.py` & `notifier.py`)
- **What it does:** Runs the entire DAG (Directed Acyclic Graph) and dispatches a rich Markdown webhook to Discord.

---

### 3. Advantages & Strengths
1. **Serverless & Free:** The entire pipeline runs on GitHub Actions cron jobs, completely free of cloud compute costs.
2. **Modular Architecture:** The NLP engine, RAG memory, and ML models are decoupled. You can easily swap XGBoost for LightGBM or FinBERT for a custom LLM.
3. **Avoids Lookahead Bias:** The `TimeSeriesSplit` cross-validation and explicit timezone-aware guard rails in the feature builder show high quantitative rigor.

---

### 4. Drawbacks & Risks
1. **Execution Slippage:** The model generates signals at 7:30 AM or 9:30 AM, but small caps are highly illiquid. Real-world execution will experience severe slippage compared to theoretical model returns.
2. **Yahoo Finance Instability:** While TradingView solved the screener, the `feature_builder.py` still relies on `yfinance` to pull 15-minute intraday candles. Yahoo is notorious for returning missing data or blocking IPs intraday. 
3. **Cost of Pinecone/OpenAI:** While compute is free via GitHub, heavy API usage of OpenAI embeddings and GPT-4o will eventually accrue costs as the daily universe expands.

---

### 5. Future Extensions & Roadmap
1. **Live Broker Integration:** Connect to Interactive Brokers or Alpaca via websockets for automated execution and limit-order management to fight slippage.
2. **Alternative Data Sources:** Swap `yfinance` 15m candle fetching for Polygon.io or Alpaca historical APIs to guarantee enterprise-grade data stability.
3. **Reinforcement Learning:** Upgrade the Pinecone RAG store to not just store win-rates, but use a Reinforcement Learning agent to adjust the weights of specific catalysts based on recent macro regimes (e.g., "FDA approvals are working better this month than Earnings").
4. **Short Selling:** The current model only looks for long squeeze candidates (`scale_pos_weight=5`). Train a parallel model to identify "Gap and Crap" candidates for short selling.
