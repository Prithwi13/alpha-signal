import os
import sys
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

from src.screener import get_stocks_in_play
from src.nlp_engine import score_news_headlines
from src.feature_builder import assemble_feature_matrix
from src.model_inference import predict_alpha_probability
from src.notifier import generate_morning_report, notify_empty_signals
from src.rag_store import init_pinecone, store_catalyst_event

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

def fetch_recent_news(ticker: str) -> list:
    """Fetches news from the last 48 hours for a given ticker using Finnhub."""
    if not FINNHUB_API_KEY:
        logger.warning("FINNHUB_API_KEY missing. Cannot fetch news.")
        return []
        
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2)
    
    url = f"https://finnhub.io/api/v1/company-news"
    params = {
        'symbol': ticker,
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d'),
        'token': FINNHUB_API_KEY
    }
    
    try:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        
        headlines = []
        for item in data:
            # Finnhub returns timestamp in seconds
            pub_time = datetime.fromtimestamp(item['datetime'], tz=timezone.utc)
            headlines.append({
                'ticker': ticker,
                'headline': item['headline'],
                'timestamp': pub_time
            })
        return headlines
    except Exception as e:
        logger.error(f"Failed to fetch news for {ticker}: {e}")
        return []

def main():
    is_simulation = "--simulate" in sys.argv
    logger.info(f"Starting Systematic Alpha Signal Engine (SASE) Pipeline... [Simulation: {is_simulation}]")
    
    # 1. Initialize DBs
    init_pinecone()
    
    # 2. Screening
    if is_simulation:
        logger.info("SIMULATION MODE: Injecting fake quantitative pre-market data to bypass screener.")
        # Inject realistic data that passes all strict filters
        fake_data = [{
            'ticker': 'SIM_BIO',
            'adv': 1500000.0,
            'current_volume': 500000.0,
            'gap_pct': 0.15,  # 15% Gap
            'rvol': 3.5,      # > 2.5
            'market_cap': 150_000_000,
            'price': 12.50
        }]
        stocks_in_play = pd.DataFrame(fake_data)
    else:
        logger.info("Screening dynamic universe of all US equities...")
        stocks_in_play = get_stocks_in_play(None)
    
    if stocks_in_play.empty:
        logger.info("No Stocks in Play found today.")
        notify_empty_signals()
        return
        
    logger.info(f"Found {len(stocks_in_play)} Stocks in Play.")
    
    candidates_with_nlp = []
    
    # 3. NLP Engine
    for idx, row in stocks_in_play.iterrows():
        ticker = row['ticker']
        
        if is_simulation:
            # Inject fake, highly impactful news to test Langchain & FinBERT
            news = [{
                'ticker': ticker,
                'headline': f"{ticker} Receives FDA Fast Track Designation for Revolutionary Cancer Treatment, Sending Shares Soaring",
                'timestamp': datetime.now(timezone.utc) - timedelta(minutes=15)
            }]
        else:
            news = fetch_recent_news(ticker)
        
        if not news:
            logger.info(f"No recent news for {ticker}, skipping NLP.")
            cand_dict = row.to_dict()
            cand_dict['decayed_sentiment'] = 0.0
            cand_dict['catalyst_category'] = 'OTHER'
            cand_dict['headline'] = 'No news'
            candidates_with_nlp.append(cand_dict)
            continue
            
        # Score the latest headline or batch of headlines
        latest_news = [news[0]]
        nlp_df = score_news_headlines(latest_news)
        
        cand_dict = row.to_dict()
        if not nlp_df.empty:
            cand_dict['decayed_sentiment'] = nlp_df.iloc[0]['decayed_sentiment']
            cand_dict['catalyst_category'] = nlp_df.iloc[0]['catalyst_category']
            cand_dict['headline'] = nlp_df.iloc[0]['headline']
        else:
            cand_dict['decayed_sentiment'] = 0.0
            cand_dict['catalyst_category'] = 'OTHER'
            cand_dict['headline'] = 'Scoring failed'
            
        candidates_with_nlp.append(cand_dict)
        
    # 4. Feature Builder
    logger.info("Building quantitative feature matrix...")
    features_df = assemble_feature_matrix(candidates_with_nlp)
    
    if features_df.empty:
        logger.info("Feature builder returned empty matrix.")
        notify_empty_signals()
        return
        
    # 5. Model Inference
    logger.info("Running XGBoost alpha predictions...")
    predictions_df = predict_alpha_probability(features_df)
    
    # 6. Logging the Results for Deep Analysis
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/sase_run_{timestamp}.csv"
    predictions_df.to_csv(log_file, index=False)
    logger.info(f"💾 Step-by-step AI Analysis saved to: {log_file}")
    
    # 7. Store significant events for future RAG (Run post-market ideally, but we store here)
    for idx, row in predictions_df.iterrows():
        # Store if we had a specific catalyst
        cand_data = next((c for c in candidates_with_nlp if c['ticker'] == row['ticker']), None)
        if cand_data and cand_data['catalyst_category'] != 'OTHER':
            store_catalyst_event(
                ticker=row['ticker'],
                catalyst_enum=cand_data['catalyst_category'],
                headline=cand_data['headline'],
                return_1h=0.0, # We don't know the future returns yet at 7:30 AM
                return_4h=0.0
            )
    
    # 8. Notify
    logger.info("Generating morning report...")
    if is_simulation:
        # In simulation, we force a high probability so the Discord webhook always fires
        predictions_df['pred_prob'] = 0.99
    generate_morning_report(predictions_df)
    logger.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()

