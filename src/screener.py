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
    Uses bulk download to avoid API rate limits (429 errors).
    """
    candidates = []
    
    if universe_list is None:
        logger.info("No universe provided. Fetching dynamically from Finnhub...")
        universe_list = get_dynamic_universe()
        logger.info(f"Dynamically gathered {len(universe_list)} tickers to screen.")
        
    # To avoid memory / timeout issues, we cap the daily scan to 800 random tickers
    batch = universe_list[:800]
    
    logger.info(f"Bulk downloading daily history for {len(batch)} tickers...")
    try:
        daily_data = yf.download(batch, period="1mo", interval="1d", group_by="ticker", threads=True, progress=False)
        intraday_data = yf.download(batch, period="1d", interval="1m", prepost=True, group_by="ticker", threads=True, progress=False)
    except Exception as e:
        logger.error(f"Bulk download failed: {e}")
        return pd.DataFrame()

    pre_filtered_tickers = []
    
    # 1. Filter by Gap and RVOL using the downloaded data
    for ticker_symbol in batch:
        try:
            # yfinance bulk download structure depends on if 1 or multiple tickers are passed
            if len(batch) == 1:
                ticker_daily = daily_data
                ticker_intra = intraday_data
            else:
                if ticker_symbol not in daily_data.columns.levels[0]:
                    continue
                ticker_daily = daily_data[ticker_symbol].dropna(how='all')
                ticker_intra = intraday_data[ticker_symbol].dropna(how='all')
                
            if len(ticker_daily) < 2 or ticker_intra.empty:
                continue
                
            hist_10d = ticker_daily.tail(10)
            adv = hist_10d['Volume'].mean()
            
            if adv == 0 or pd.isna(adv):
                continue
                
            prev_close = ticker_daily['Close'].iloc[-2]
            current_bar_volume = ticker_intra['Volume'].sum()
            open_today = ticker_intra['Open'].iloc[0]
            
            gap_pct = (open_today - prev_close) / prev_close
            rvol = current_bar_volume / (adv / 39)
            
            if abs(gap_pct) >= 0.03 and rvol >= 2.5:
                pre_filtered_tickers.append({
                    'ticker': ticker_symbol,
                    'adv': adv,
                    'current_volume': current_bar_volume,
                    'gap_pct': gap_pct,
                    'rvol': rvol
                })
        except Exception:
            continue
            
    logger.info(f"Pre-filter complete. Found {len(pre_filtered_tickers)} candidates. Checking Market Cap...")
    
    # 2. Only check `.info` for the few stocks that passed the pre-filter
    for cand in pre_filtered_tickers:
        ticker_symbol = cand['ticker']
        try:
            ticker = yf.Ticker(ticker_symbol)
            # Use fast_info to avoid HTTP 429 if possible, fallback to info
            try:
                market_cap = ticker.fast_info['marketCap']
                current_price = ticker.fast_info['lastPrice']
            except:
                info = ticker.info
                market_cap = info.get('marketCap', 0)
                current_price = info.get('regularMarketPrice') or info.get('currentPrice', 0)
                
            if not (50_000_000 <= market_cap <= 2_000_000_000):
                continue
            if not (1.00 <= current_price <= 25.00):
                continue
                
            cand['market_cap'] = market_cap
            cand['price'] = current_price
            candidates.append(cand)
            
        except Exception as e:
            logger.debug(f"Failed to process info for {ticker_symbol}: {e}")

    df = pd.DataFrame(candidates)
    if df.empty:
        return df
        
    df = df.sort_values('rvol', ascending=False).head(15).reset_index(drop=True)
    return df
