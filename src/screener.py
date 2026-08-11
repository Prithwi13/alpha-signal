import pandas as pd
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_stocks_in_play(universe_list: list[str] = None) -> pd.DataFrame:
    """
    Sifts through the US market using the free, unauthenticated TradingView Scanner API.
    This completely bypasses the GitHub Actions rate limits of yfinance.
    Finds up to 15 "Stocks in Play" based on Gap % and RVOL.
    """
    logger.info("Connecting to TradingView Scanner API...")
    url = "https://scanner.tradingview.com/america/scan"
    
    # We query the entire US equity market (NASDAQ, NYSE, AMEX)
    payload = {
        "filter": [
            {"left": "type", "operation": "in_range", "right": ["stock", "dr", "fund"]},
            {"left": "exchange", "operation": "in_range", "right": ["AMEX", "NASDAQ", "NYSE"]},
            # Ensure they have pre-market volume
            {"left": "premarket_volume", "operation": "nempty"}
        ],
        "options": {"lang": "en"},
        "markets": ["america"],
        "symbols": {"query": {"types": []}, "tickers": []},
        # We request exactly the data we need to calculate Gap and RVOL
        "columns": [
            "name", 
            "premarket_change", 
            "premarket_volume", 
            "market_cap_basic", 
            "average_volume_10d_calc",
            "close" # For price filtering
        ],
        "sort": {"sortBy": "premarket_change", "sortOrder": "desc"},
        "range": [0, 500] # Get the top 500 gappers to filter down
    }
    
    candidates = []
    
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json().get('data', [])
        logger.info(f"Retrieved {len(data)} pre-market movers from TradingView.")
        
        for item in data:
            ticker_symbol = item['d'][0]
            premarket_change_pct = item['d'][1]
            premarket_volume = item['d'][2]
            market_cap = item['d'][3]
            adv_10d = item['d'][4]
            current_price = item['d'][5]
            
            # Handle nulls
            if None in (premarket_change_pct, premarket_volume, market_cap, adv_10d, current_price):
                continue
                
            # Filter 1: Gap % > 3%
            gap_pct = premarket_change_pct / 100.0
            if abs(gap_pct) < 0.03:
                continue
                
            # Filter 2: Price between $1 and $25
            if not (1.00 <= current_price <= 25.00):
                continue
                
            # Filter 3: Market Cap between $50M and $2B
            if not (50_000_000 <= market_cap <= 2_000_000_000):
                continue
                
            # Filter 4: RVOL > 2.5
            if adv_10d == 0:
                continue
            rvol = premarket_volume / (adv_10d / 39) # Normalizing ADV to typical 390 min trading day chunks (39 * 10min)
            
            if rvol >= 2.5:
                candidates.append({
                    'ticker': ticker_symbol,
                    'adv': adv_10d,
                    'current_volume': premarket_volume,
                    'gap_pct': gap_pct,
                    'rvol': rvol,
                    'market_cap': market_cap,
                    'price': current_price
                })
                
    except Exception as e:
        logger.error(f"Failed to fetch data from TradingView: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(candidates)
    if df.empty:
        logger.warning("No stocks passed the strict quantitative filters today.")
        return df
        
    df = df.sort_values('rvol', ascending=False).head(15).reset_index(drop=True)
    logger.info(f"Found {len(df)} Stocks in Play!")
    return df

