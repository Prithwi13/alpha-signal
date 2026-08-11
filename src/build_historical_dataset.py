import os
import time
import logging
import pandas as pd
from datetime import datetime, timedelta, timezone
import yfinance as yf
from src.nlp_engine import score_news_headlines
from src.feature_builder import build_features
from src.model_trainer import train_and_evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def gather_historical_data(universe: list, days_back: int = 30) -> pd.DataFrame:
    """
    Crawls historical data and news for the universe to build a realistic training dataset.
    """
    import requests
    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
    
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)
    
    all_features = []
    
    logger.info(f"Gathering real historical data for {len(universe)} tickers over {days_back} days...")
    
    for ticker in universe:
        logger.info(f"Processing {ticker}...")
        
        # 1. Fetch News
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
            news_data = resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch news for {ticker}: {e}")
            continue
            
        if not news_data:
            continue
            
        # Parse news items
        headlines = []
        for item in news_data:
            pub_time = datetime.fromtimestamp(item['datetime'], tz=timezone.utc)
            headlines.append({
                'ticker': ticker,
                'headline': item['headline'],
                'timestamp': pub_time
            })
            
        if not headlines:
            continue
            
        # 2. Score news via NLP engine
        # Warning: This calls OpenAI for each headline > 0.4. To save tokens in backtest, 
        # we might just do this on a sample or limit to top 50. Let's limit for the crawler.
        headlines = headlines[:20] 
        nlp_df = score_news_headlines(headlines)
        
        if nlp_df.empty:
            continue
            
        # 3. For each news event, simulate execution 2 hours later to gather point-in-time features
        for idx, row in nlp_df.iterrows():
            event_time = row['timestamp']
            
            # Guard time: The moment we would run the model (e.g. 2 hours after news)
            guard_time = event_time + timedelta(hours=2)
            
            # Ensure guard_time is within trading hours or just use it raw (yfinance handles this mostly)
            # but yfinance limits 15m data to 60 days max. If guard time is > 60 days, it fails.
            if guard_time < (end_date - timedelta(days=59)):
                continue
                
            nlp_data = {
                'decayed_sentiment': row['decayed_sentiment'],
                'catalyst_category': row['catalyst_category']
            }
            
            # 4. Build point-in-time features
            features = build_features(ticker, nlp_data, lookahead_guard_time=guard_time)
            
            if not features:
                continue
                
            # 5. Fetch Target Variable (Excess Return in next 2 hours)
            # We need price at guard_time, and price at guard_time + 2 hours
            stock = yf.Ticker(ticker)
            iwm = yf.Ticker("IWM")
            
            target_time = guard_time + timedelta(hours=2)
            
            try:
                hist_15m = stock.history(start=guard_time.strftime('%Y-%m-%d'), end=(target_time + timedelta(days=1)).strftime('%Y-%m-%d'), interval="15m")
                iwm_15m = iwm.history(start=guard_time.strftime('%Y-%m-%d'), end=(target_time + timedelta(days=1)).strftime('%Y-%m-%d'), interval="15m")
                
                # Align timezones
                if hist_15m.index.tz is None: hist_15m.index = hist_15m.index.tz_localize('UTC')
                if iwm_15m.index.tz is None: iwm_15m.index = iwm_15m.index.tz_localize('UTC')
                
                # Filter rows between guard and target
                hist_window = hist_15m[(hist_15m.index >= guard_time) & (hist_15m.index <= target_time)]
                iwm_window = iwm_15m[(iwm_15m.index >= guard_time) & (iwm_15m.index <= target_time)]
                
                if len(hist_window) < 2 or len(iwm_window) < 2:
                    continue
                    
                p0 = hist_window['Open'].iloc[0]
                p1 = hist_window['Close'].iloc[-1]
                ret = (p1 - p0) / p0
                
                iwm_p0 = iwm_window['Open'].iloc[0]
                iwm_p1 = iwm_window['Close'].iloc[-1]
                iwm_ret = (iwm_p1 - iwm_p0) / iwm_p0
                
                beta = features.get('sector_beta', 1.0)
                excess_ret = ret - (beta * iwm_ret)
                
                features['timestamp'] = guard_time
                features['target_class'] = 1 if excess_ret > 0.03 else 0
                all_features.append(features)
                
            except Exception as e:
                logger.error(f"Error computing target for {ticker} at {guard_time}: {e}")
                continue
                
        time.sleep(1) # Finnhub rate limit
        
    df = pd.DataFrame(all_features)
    return df

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    logger.info("Starting historical data crawler...")
    from src.screener import get_dynamic_universe
    
    universe = get_dynamic_universe()
    # We slice to a manageable number for the backtester crawler to avoid rate limiting
    universe = universe[:50]
    
    df = gather_historical_data(universe, days_back=10)
    
    if not df.empty:
        logger.info(f"Gathered {len(df)} historical point-in-time samples.")
        df = df.sort_values('timestamp')
        df.to_csv("data/historical_features.csv", index=False)
        
        logger.info("Training models on real historical data...")
        train_and_evaluate(df, target_col="target_class")
    else:
        logger.warning("Crawler found no valid overlapping data points.")
