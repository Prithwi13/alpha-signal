import os
import joblib
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import xgboost as xgb

logger = logging.getLogger(__name__)

MODELS_DIR = "models"

def train_and_evaluate(df: pd.DataFrame, target_col: str = "target_class"):
    """
    Evaluates multiple baseline models using TimeSeriesSplit and saves the best model.
    Assumes df is sorted by time.
    """
    if df.empty or len(df) < 50:
        logger.warning("Not enough data to train models.")
        return None
        
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Exclude non-feature columns
    exclude_cols = ['ticker', 'timestamp', target_col]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols]
    y = df[target_col]
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    models = {
        "LogisticRegression": Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(class_weight='balanced', max_iter=1000))
        ]),
        "LightGBM": lgb.LGBMClassifier(
            max_depth=3,
            learning_rate=0.03,
            n_estimators=150,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=42,
            verbosity=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            max_depth=3,
            learning_rate=0.03,
            n_estimators=150,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            scale_pos_weight=5, # Approximate class balancing
            random_state=42
        )
    }
    
    results = {name: {'precision': [], 'recall': [], 'roc_auc': []} for name in models}
    
    logger.info(f"Starting TimeSeries Cross Validation on {len(X)} samples with {len(feature_cols)} features...")
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Check if fold has both classes
        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            logger.warning(f"Fold {fold} missing classes. Skipping.")
            continue
            
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            probs = model.predict_proba(X_val)[:, 1]
            
            p = precision_score(y_val, preds, zero_division=0)
            r = recall_score(y_val, preds, zero_division=0)
            auc = roc_auc_score(y_val, probs)
            
            results[name]['precision'].append(p)
            results[name]['recall'].append(r)
            results[name]['roc_auc'].append(auc)
            
    # Aggregate and print
    best_model_name = "XGBoost" # default
    best_prec = 0.0
    
    print("\n--- Model Comparison Results ---")
    for name in models.keys():
        avg_p = np.mean(results[name]['precision']) if results[name]['precision'] else 0
        avg_r = np.mean(results[name]['recall']) if results[name]['recall'] else 0
        avg_auc = np.mean(results[name]['roc_auc']) if results[name]['roc_auc'] else 0
        
        print(f"{name} -> Precision: {avg_p:.4f} | Recall: {avg_r:.4f} | ROC-AUC: {avg_auc:.4f}")
        
        if avg_p > best_prec:
            best_prec = avg_p
            best_model_name = name
            
    print(f"\nSelected {best_model_name} as the final production model (optimizing for Precision).")
    
    # Train best model on ALL data for production
    final_model = models[best_model_name]
    final_model.fit(X, y)
    
    model_path = os.path.join(MODELS_DIR, "production_alpha_model.pkl")
    joblib.dump(final_model, model_path)
    # Also save a copy as xgboost_alpha.pkl just to satisfy hardcoded references if any
    joblib.dump(final_model, os.path.join(MODELS_DIR, "xgboost_alpha.pkl"))
    
    logger.info(f"Model saved to {model_path}")
    
    # Return feature names for inference ordering
    return feature_cols

if __name__ == "__main__":
    # Dummy run if executed directly
    logger.info("Executing model_trainer module.")
