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
        
    # We will bulk download the ENTIRE universe (3000+ tickers).
    # To avoid triggering Yahoo Finance's DDoS protection (which causes the "Expecting value" error),
    # we chunk the requests and use a realistic User-Agent.
    batch = universe_list
    
    import time
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    
    logger.info(f"Chunk downloading daily history for the entire market ({len(batch)} tickers)...")
    
    daily_data_frames = []
    intraday_data_frames = []
    chunk_size = 200
    
    try:
        for i in range(0, len(batch), chunk_size):
            chunk = batch[i:i+chunk_size]
            logger.info(f"Downloading chunk {i//chunk_size + 1}/{(len(batch)//chunk_size) + 1}...")
            
            # Using threads=5 instead of True to prevent flooding
            d = yf.download(chunk, period="1mo", interval="1d", group_by="ticker", threads=5, session=session, progress=False)
            i_d = yf.download(chunk, period="1d", interval="1m", prepost=True, group_by="ticker", threads=5, session=session, progress=False)
            
            if not d.empty: daily_data_frames.append(d)
            if not i_d.empty: intraday_data_frames.append(i_d)
            
            time.sleep(2) # Brief pause between chunks to respect rate limits
            
        if daily_data_frames and intraday_data_frames:
            daily_data = pd.concat(daily_data_frames, axis=1)
            intraday_data = pd.concat(intraday_data_frames, axis=1)
        else:
            return pd.DataFrame()
            
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
    
    # 2. Only check Market Cap using Finnhub for the few stocks that passed the pre-filter
    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
    import time
    
    for cand in pre_filtered_tickers:
        ticker_symbol = cand['ticker']
        try:
            # We already have price from our intraday download
            current_price = cand['current_volume'] # wait, need price, we will just use intraday close
            # We didn't save current price in cand, let's just fetch it from Finnhub quote or use Finnhub profile2
            # Finnhub profile2 gives market cap in millions
            url = f"https://finnhub.io/api/v1/stock/profile2"
            params = {'symbol': ticker_symbol, 'token': FINNHUB_API_KEY}
            resp = requests.get(url, params=params)
            
            if resp.status_code == 429:
                logger.warning("Finnhub rate limit hit. Sleeping for 1 minute.")
                time.sleep(60)
                resp = requests.get(url, params=params)
                
            resp.raise_for_status()
            profile_data = resp.json()
            
            # marketCapitalization is in Millions of USD
            market_cap_millions = profile_data.get('marketCapitalization', 0)
            market_cap = market_cap_millions * 1_000_000
            
            # Fetch current price via Finnhub quote to be safe, or just use the YF price we had
            quote_url = f"https://finnhub.io/api/v1/quote"
            quote_resp = requests.get(quote_url, params=params)
            if quote_resp.status_code == 200:
                current_price = quote_resp.json().get('c', 0)
            else:
                current_price = 0
                
            if not (50_000_000 <= market_cap <= 2_000_000_000):
                continue
            if not (1.00 <= current_price <= 25.00):
                continue
                
            cand['market_cap'] = market_cap
            cand['price'] = current_price
            candidates.append(cand)
            time.sleep(1) # Finnhub free tier limit is 60 calls/min
            
        except Exception as e:
            logger.debug(f"Failed to process info for {ticker_symbol}: {e}")

    df = pd.DataFrame(candidates)
    if df.empty:
        return df
        
    df = df.sort_values('rvol', ascending=False).head(15).reset_index(drop=True)
    return df
