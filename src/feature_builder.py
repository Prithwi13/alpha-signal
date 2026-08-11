import pandas as pd
import numpy as pd_np
import numpy as np
import yfinance as yf
import logging
from typing import List, Dict, Any
from src.nlp_engine import CatalystType
from src.rag_store import query_catalyst_history

logger = logging.getLogger(__name__)

def calc_rsi(series: pd.Series, periods: int = 14) -> pd.Series:
    delta = series.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up1 = up.ewm(span=periods).mean()
    roll_down1 = down.abs().ewm(span=periods).mean()
    RS1 = roll_up1 / roll_down1
    RSI1 = 100.0 - (100.0 / (1.0 + RS1))
    return RSI1

def calculate_sector_beta(ticker_returns: pd.Series, benchmark_returns: pd.Series, window: int = 30) -> float:
    """Calculate rolling Beta relative to IWM over the given window."""
    if len(ticker_returns) < 2 or len(benchmark_returns) < 2:
        return 1.0 # Default Beta
        
    cov = ticker_returns.cov(benchmark_returns)
    var = benchmark_returns.var()
    if var == 0 or np.isnan(var):
        return 1.0
    return cov / var

def build_features(
    ticker: str, 
    nlp_data: dict, 
    lookahead_guard_time: pd.Timestamp = None
) -> dict:
    """
    Builds the final feature row for a stock.
    nlp_data: {'decayed_sentiment': float, 'catalyst_category': str}
    """
    try:
        # Fetch 15m data for the stock
        stock = yf.Ticker(ticker)
        # 15m data max period is 60d
        hist_15m = stock.history(period="5d", interval="15m")
        
        # Fetch IWM (benchmark) 15m data
        iwm = yf.Ticker("IWM")
        iwm_15m = iwm.history(period="5d", interval="15m")
        
        # Lookahead Guard: Ensure no data past guard time is used
        if lookahead_guard_time:
            # Timezone handling
            if hist_15m.index.tz is None:
                hist_15m.index = hist_15m.index.tz_localize('UTC')
            if iwm_15m.index.tz is None:
                iwm_15m.index = iwm_15m.index.tz_localize('UTC')
                
            # Filter strictly BEFORE or AT guard time
            hist_15m = hist_15m[hist_15m.index <= lookahead_guard_time]
            iwm_15m = iwm_15m[iwm_15m.index <= lookahead_guard_time]
            
            if len(hist_15m) == 0:
                logger.warning(f"[{ticker}] No data available before guard time {lookahead_guard_time}. Possible lookahead leak prevented.")
                return {}
                
        if len(hist_15m) < 15: # Need enough data for RSI and momentum
            logger.warning(f"[{ticker}] Not enough 15m candles to build features.")
            return {}

        # Align indexes
        hist_15m, iwm_15m = hist_15m.align(iwm_15m, join='inner', axis=0)

        # 1. rvol_15m (15-minute Relative Volume)
        # Using 5-day average volume for the same 15-minute bucket could be complex.
        # Approximation: current volume / average 15m volume over last 5 days
        avg_15m_vol = hist_15m['Volume'].iloc[:-1].mean()
        curr_15m_vol = hist_15m['Volume'].iloc[-1]
        rvol_15m = curr_15m_vol / avg_15m_vol if avg_15m_vol > 0 else 1.0
        
        # 2. momentum_1h (P_t / P_{t-4} - 1)
        # 4 bars of 15m = 1 hour
        p_t = hist_15m['Close'].iloc[-1]
        p_t_4 = hist_15m['Close'].iloc[-5] if len(hist_15m) >= 5 else hist_15m['Close'].iloc[0]
        momentum_1h = (p_t / p_t_4) - 1.0
        
        # IWM 1h momentum
        iwm_p_t = iwm_15m['Close'].iloc[-1]
        iwm_p_t_4 = iwm_15m['Close'].iloc[-5] if len(iwm_15m) >= 5 else iwm_15m['Close'].iloc[0]
        iwm_momentum_1h = (iwm_p_t / iwm_p_t_4) - 1.0

        # 3. rsi_14 (14-period RSI on 15m candles)
        rsi_series = calc_rsi(hist_15m['Close'], periods=14)
        rsi_14 = rsi_series.iloc[-1]
        
        # 4. sector_beta (30-day rolling Beta)
        # We need daily data for a proper 30-day beta
        hist_daily = stock.history(period="2mo", interval="1d")
        iwm_daily = iwm.history(period="2mo", interval="1d")
        
        if lookahead_guard_time:
            # We must use guard time's date for daily cutoff to prevent leak
            cutoff_date = pd.to_datetime(lookahead_guard_time.date())
            # yfinance index tz handling
            if hist_daily.index.tz is not None:
                cutoff_date_tz = cutoff_date.tz_localize(hist_daily.index.tz)
            else:
                cutoff_date_tz = cutoff_date
            
            hist_daily = hist_daily[hist_daily.index < cutoff_date_tz] # Strict less than to only use PAST days
            
            if iwm_daily.index.tz is not None:
                cutoff_date_iwm = cutoff_date.tz_localize(iwm_daily.index.tz)
            else:
                cutoff_date_iwm = cutoff_date
            iwm_daily = iwm_daily[iwm_daily.index < cutoff_date_iwm]
            
        ticker_returns = hist_daily['Close'].pct_change().dropna()
        bm_returns = iwm_daily['Close'].pct_change().dropna()
        ticker_returns, bm_returns = ticker_returns.align(bm_returns, join='inner')
        
        # Use last 30 days
        beta_30d = calculate_sector_beta(ticker_returns.tail(30), bm_returns.tail(30))
        
        # 5. excess_momentum
        excess_momentum = momentum_1h - (beta_30d * iwm_momentum_1h)
        
        # 6. RAG History
        cat_enum = nlp_data.get('catalyst_category', 'OTHER')
        rag_metrics = query_catalyst_history(ticker, cat_enum)
        rag_historical_win_rate = rag_metrics.get('win_rate', 0.5)
        
        # 7. One-hot encoding categories manually for the dict
        # In a real pipeline, sklearn's OneHotEncoder is used, but dict is easier for single rows
        cats = [e.value for e in CatalystType]
        encoded_cats = {f"cat_{c}": 1 if c == cat_enum else 0 for c in cats}
        
        feature_dict = {
            "ticker": ticker,
            "rvol_15m": float(rvol_15m),
            "momentum_1h": float(momentum_1h),
            "rsi_14": float(rsi_14),
            "decayed_sentiment": float(nlp_data.get('decayed_sentiment', 0.0)),
            "rag_historical_win_rate": float(rag_historical_win_rate),
            "sector_beta": float(beta_30d),
            "excess_momentum": float(excess_momentum)
        }
        
        feature_dict.update(encoded_cats)
        
        # Clean NaNs
        for k, v in feature_dict.items():
            if isinstance(v, float) and np.isnan(v):
                feature_dict[k] = 0.0
                
        return feature_dict
        
    except Exception as e:
        logger.error(f"Failed to build features for {ticker}: {e}")
        return {}

def assemble_feature_matrix(candidates: List[dict], lookahead_guard_time=None) -> pd.DataFrame:
    """Takes a list of dicts with ticker and nlp info and builds the feature matrix."""
    rows = []
    for cand in candidates:
        ticker = cand['ticker']
        nlp_data = {
            'decayed_sentiment': cand.get('decayed_sentiment', 0.0),
            'catalyst_category': cand.get('catalyst_category', 'OTHER')
        }
        features = build_features(ticker, nlp_data, lookahead_guard_time)
        if features:
            rows.append(features)
            
    return pd.DataFrame(rows)
