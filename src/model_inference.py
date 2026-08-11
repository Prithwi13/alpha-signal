import os
import joblib
import pandas as pd
import logging

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join("models", "production_alpha_model.pkl")
FALLBACK_MODEL_PATH = os.path.join("models", "xgboost_alpha.pkl")

def load_model():
    """Loads the pre-trained model."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    elif os.path.exists(FALLBACK_MODEL_PATH):
        return joblib.load(FALLBACK_MODEL_PATH)
    else:
        logger.error("No trained model found.")
        return None

def predict_alpha_probability(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame of features and outputs prediction class and P(Class=1).
    """
    model = load_model()
    if not model or features_df.empty:
        features_df['pred_class'] = 0
        features_df['pred_prob'] = 0.0
        return features_df
        
    # We must drop non-feature columns just like in training
    exclude_cols = ['ticker', 'timestamp', 'target_class']
    feature_cols = [c for c in features_df.columns if c not in exclude_cols]
    
    # Add the engineered features exactly as they were in quant_pipeline.py
    if not features_df.empty:
        if 'rvol_15m' in features_df.columns and 'momentum_1h' in features_df.columns:
            features_df['vol_momentum_interaction'] = features_df['rvol_15m'] * features_df['momentum_1h']
            feature_cols.append('vol_momentum_interaction')
        if 'decayed_sentiment' in features_df.columns and 'momentum_1h' in features_df.columns:
            features_df['sentiment_momentum_interaction'] = features_df['decayed_sentiment'] * features_df['momentum_1h']
            feature_cols.append('sentiment_momentum_interaction')
    
    # Fill any remaining NaNs with 0
    features_df = features_df.fillna(0.0)
    
    # Ensure exact column order and presence based on what the model learned
    if hasattr(model, 'feature_names_in_'):
        expected_cols = model.feature_names_in_
        for col in expected_cols:
            if col not in features_df.columns:
                features_df[col] = 0.0
        X = features_df[expected_cols]
    else:
        X = features_df[feature_cols]
    
    try:
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1] # Probability of class 1
        
        features_df['pred_class'] = preds
        features_df['pred_prob'] = probs
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        features_df['pred_class'] = 0
        features_df['pred_prob'] = 0.0
        
    return features_df
