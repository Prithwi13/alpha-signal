import pandas as pd
from src.screener import get_stocks_in_play
import logging

logging.basicConfig(level=logging.INFO)

def test_screener():
    # A mix of typical small/mid cap tickers, some of which might be volatile today
    sample_universe = [
        "SENS", "OCGN", "TNXP", "ZOM", "JAGX", "CTRM", "SNDL", "BNGO",
        "NAKD", "GME", "AMC", "BB", "KOSS", "EXPR", "RKLB", "SOUN",
        "PLTR", "IONQ", "LCID", "ACHR", "JOBY", "ASTS", "HIMS"
    ]
    
    print(f"Running screener on universe of {len(sample_universe)} tickers...")
    df = get_stocks_in_play(sample_universe)
    
    print("\n--- Stocks in Play ---")
    if df.empty:
        print("No stocks passed the strict momentum filters today.")
    else:
        print(df.to_string())

if __name__ == "__main__":
    test_screener()
