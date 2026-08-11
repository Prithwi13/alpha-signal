import pandas as pd
import numpy as np
import os
from src.model_trainer import train_and_evaluate
from src.nlp_engine import CatalystType

n_samples = 1000
np.random.seed(42)

# Same features as feature_builder.py
df = pd.DataFrame({
    'ticker': ['DUMMY'] * n_samples,
    'timestamp': pd.date_range(start='1/1/2026', periods=n_samples, freq='D'),
    'rvol_15m': np.random.uniform(1.0, 15.0, n_samples),
    'momentum_1h': np.random.uniform(-0.1, 0.2, n_samples),
    'rsi_14': np.random.uniform(20, 90, n_samples),
    'decayed_sentiment': np.random.uniform(-1.0, 1.0, n_samples),
    'rag_historical_win_rate': np.random.uniform(0.3, 0.8, n_samples),
    'sector_beta': np.random.uniform(0.5, 2.0, n_samples),
    'excess_momentum': np.random.uniform(-0.05, 0.1, n_samples)
})

cats = [e.value for e in CatalystType]
for c in cats:
    df[f"cat_{c}"] = np.random.randint(0, 2, n_samples)

# Artificial target class
target_prob = (df['momentum_1h'] * 0.3 + df['decayed_sentiment'] * 0.2 + df['rag_historical_win_rate'] * 0.4)
target_prob = (target_prob - target_prob.min()) / (target_prob.max() - target_prob.min())
df['target_class'] = (target_prob > 0.7).astype(int)

# This will generate models/production_alpha_model.pkl
train_and_evaluate(df, "target_class")

