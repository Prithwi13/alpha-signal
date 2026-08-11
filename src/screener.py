import pandas as pd
import yfinance as yf
import logging
import requests
import os
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_dynamic_universe() -> list[str]:
    """
    Fetches all US common stocks dynamically from Finnhub.
    Returns a list of ticker symbols.
    """
    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
    if not FINNHUB_API_KEY:
        logger.warning("No Finnhub API key found. Using default list.")
        return ["SENS", "OCGN", "TNXP", "ZOM", "JAGX"]
        
    url = "https://finnhub.io/api/v1/stock/symbol"
    params = {'exchange': 'US', 'token': FINNHUB_API_KEY}
    
    try:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        
        # Filter for common stocks
        tickers = [item['symbol'] for item in data if item.get('type') == 'Common Stock']
        
        # We shuffle and return up to 3000 tickers to avoid massive runtimes
        random.seed(datetime.now().timestamp()) if 'datetime' in globals() else random.seed()
        random.shuffle(tickers)
        return tickers[:3000]
        
    except Exception as e:
        logger.error(f"Failed to fetch dynamic universe: {e}")
        return ["SENS", "OCGN", "TNXP", "ZOM", "JAGX"]

def get_stocks_in_play(universe_list: list[str] = None) -> pd.DataFrame:
    """
    Sifts through a universe of tickers to find 15 "Stocks in Play".
    If universe_list is None, fetches it dynamically.
    
    Filters:
    - Market Cap: $50M to $2B
    - Price: $1.00 to $25.00
    - Gap Percentage: >= 3% or <= -3%
    - RVOL: >= 2.5
    """
    candidates = []
    
    if universe_list is None:
        logger.info("No universe provided. Fetching dynamically from Finnhub...")
        universe_list = get_dynamic_universe()
        logger.info(f"Dynamically gathered {len(universe_list)} tickers to screen.")
        
    for ticker_symbol in universe_list:
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            
            if not info:
                continue
                
            # Filter 1: Market Cap between 50M and 2B
            market_cap = info.get('marketCap', 0)
            if not (50_000_000 <= market_cap <= 2_000_000_000):
                continue
                
            # Filter 2: Price between 1.00 and 25.00
            current_price = info.get('regularMarketPrice') or info.get('currentPrice')
            if current_price is None or not (1.00 <= current_price <= 25.00):
                continue
                
            # Fetch last 10 days of data to compute ADV and previous close
            # Using '1d' interval
            hist_daily = ticker.history(period="1mo", interval="1d")
            if hist_daily.empty or len(hist_daily) < 2:
                continue
                
            hist_10d = hist_daily.tail(10)
            adv = hist_10d['Volume'].mean()
            
            if adv == 0 or pd.isna(adv):
                continue
                
            # Get previous close
            prev_close = hist_daily['Close'].iloc[-2]
            
            # Get today's open and current volume from intraday or daily data
            # Since this runs at 7:30 AM EST (pre-market), we might need to rely on pre-market data if available,
            # or the latest available info. YFinance can fetch pre-market with include_prepost=True
            hist_intraday = ticker.history(period="1d", interval="1m", prepost=True)
            if hist_intraday.empty:
                continue
                
            # Today's open is the first print of today's session (which includes pre-market)
            # Actually, standard Gap is usually Regular Trading Hours (RTH) open vs prev close,
            # but since it's 7:30 AM, we use the current pre-market price as "Open_today" for the gap calc,
            # or the actual first pre-market trade. Let's use the most recent price.
            current_bar_volume = hist_intraday['Volume'].sum() # Total volume so far today
            open_today = hist_intraday['Open'].iloc[0] # First trade today
            
            # Gap Percentage
            gap_pct = (open_today - prev_close) / prev_close
            
            # RVOL
            rvol = current_bar_volume / (adv / 39)
            
            # Filter 3: Gap >= 3% and RVOL >= 2.5
            if abs(gap_pct) >= 0.03 and rvol >= 2.5:
                candidates.append({
                    'ticker': ticker_symbol,
                    'market_cap': market_cap,
                    'price': current_price,
                    'adv': adv,
                    'current_volume': current_bar_volume,
                    'gap_pct': gap_pct,
                    'rvol': rvol
                })
                
        except Exception as e:
            logger.debug(f"Failed to process {ticker_symbol}: {e}")
            
    df = pd.DataFrame(candidates)
    if df.empty:
        return df
        
    # Sort by RVOL descending and cap at 15
    df = df.sort_values('rvol', ascending=False).head(15).reset_index(drop=True)
    return df
