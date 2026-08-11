import os
import requests
import logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

def send_to_discord(payload: dict):
    if not DISCORD_WEBHOOK_URL:
        logger.warning("DISCORD_WEBHOOK_URL not set. Skipping Discord notification.")
        return
        
    try:
        resp = requests.post(DISCORD_WEBHOOK_URL, json=payload)
        resp.raise_for_status()
        logger.info("Discord notification sent successfully.")
    except Exception as e:
        logger.error(f"Failed to send Discord notification: {e}")

def notify_empty_signals():
    payload = {
        "content": "📊 **Morning Alpha Report**\n\nNo small-cap stocks passed the strict momentum and quantitative filters today. Staying flat.",
        "username": "SASE Quant Bot"
    }
    send_to_discord(payload)

def generate_morning_report(predictions_df: pd.DataFrame):
    """
    Takes the dataframe of predictions, filters for P(Class=1) >= 0.70,
    and sends a formatted Markdown alert to Discord.
    """
    if predictions_df.empty:
        notify_empty_signals()
        return
        
    # Filter for high probability signals
    signals = predictions_df[predictions_df['pred_prob'] >= 0.70]
    
    if signals.empty:
        notify_empty_signals()
        return
        
    # Sort by probability
    signals = signals.sort_values(by='pred_prob', ascending=False)
    
    report_lines = ["📊 **QUANT ALPHA SIGNAL REPORT** 📊\n"]
    report_lines.append(f"*Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EST*\n")
    
    for _, row in signals.iterrows():
        ticker = row['ticker']
        prob = row['pred_prob'] * 100
        sentiment = row.get('decayed_sentiment', 0.0)
        
        # Recover catalyst from one-hot encoding if present, else fallback
        catalyst = "OTHER"
        for col in row.index:
            if col.startswith('cat_') and row[col] == 1:
                catalyst = col.replace('cat_', '')
                break
                
        rvol = row.get('rvol_15m', 0.0)
        win_rate = row.get('rag_historical_win_rate', 0.5) * 100
        
        entry = f"""**Actionable Signal: STRONG BULLISH**
**${ticker}** - {prob:.1f}% Probability of >3% excess return.
- **Catalyst Engine:** {catalyst} (FinBERT Decayed Sentiment: {sentiment:.2f})
- **RAG Historical Precedence:** {win_rate:.1f}% win-rate for similar small-cap catalysts.
- **Quant Math:** 15m RVOL is {rvol:.1f}x average.
"""
        report_lines.append(entry)
        
    report_lines.append("\n_Note: Trade strictly according to dynamic risk models._")
    
    payload = {
        "content": "\n".join(report_lines),
        "username": "SASE Quant Bot"
    }
    
    send_to_discord(payload)
