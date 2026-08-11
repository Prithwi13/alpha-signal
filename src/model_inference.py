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
    
    X = features_df[feature_cols]
    
    # Fill any remaining NaNs with 0
    X = X.fillna(0.0)
    
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
