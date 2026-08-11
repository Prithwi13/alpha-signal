import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, roc_curve, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import xgboost as xgb

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set aesthetic style
plt.style.use('dark_background')
sns.set_palette("viridis")

ARTIFACT_DIR = "/Users/prithwirajchatterjee/.gemini/antigravity/brain/378eb12c-66b8-404a-a493-172e9d09a411"
PLOTS_DIR = os.path.join(ARTIFACT_DIR, ".user_uploaded")
os.makedirs(PLOTS_DIR, exist_ok=True)
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def run_quant_pipeline():
    logger.info("1. Loading Real Historical Dataset...")
    df = pd.read_csv("data/historical_features.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    logger.info(f"Loaded {len(df)} samples across {df['ticker'].nunique()} tickers.")
    
    logger.info("2. Feature Engineering...")
    # Add engineered interaction features
    df['vol_momentum_interaction'] = df['rvol_15m'] * df['momentum_1h']
    df['sentiment_momentum_interaction'] = df['decayed_sentiment'] * df['momentum_1h']
    
    logger.info("3. Statistical Analysis & Quant Tests...")
    # T-test for RVOL between Class 1 (winners) and Class 0 (losers)
    winners = df[df['target_class'] == 1]['rvol_15m']
    losers = df[df['target_class'] == 0]['rvol_15m']
    t_stat, p_val = stats.ttest_ind(winners, losers, equal_var=False)
    logger.info(f"T-Test RVOL (Winners vs Losers): T-Stat={t_stat:.2f}, P-Value={p_val:.4f}")
    
    # Exclude non-feature columns
    exclude_cols = ['ticker', 'timestamp', 'target_class', 'target_max_return', 'target_pct']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    logger.info("4. Exploratory Data Analysis (EDA)...")
    plt.figure(figsize=(14, 12))
    corr = df[feature_cols + ['target_class']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title("Real Historical Feature Correlation Heatmap")
    plt.tight_layout()
    corr_path = os.path.join(PLOTS_DIR, "real_corr_heatmap.png")
    plt.savefig(corr_path, dpi=150)
    plt.close()
    
    logger.info("5. Model Building & Validation without Overfitting (TimeSeriesSplit)...")
    X = df[feature_cols]
    y = df['target_class']
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    models = {
        "LogisticRegression": Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(class_weight='balanced', max_iter=1000))
        ]),
        "LightGBM": lgb.LGBMClassifier(
            max_depth=3, # Prevent overfitting
            learning_rate=0.03,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=42,
            verbosity=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            max_depth=3, # Prevent overfitting
            learning_rate=0.03,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            scale_pos_weight=sum(y==0)/sum(y==1) if sum(y==1)>0 else 1, # Imbalance handling
            random_state=42
        )
    }
    
    # We will track the ROC curve for the LAST fold as a validation proxy
    last_fold_roc = {}
    last_fold_cm = {}
    results = {name: {'precision': [], 'recall': [], 'roc_auc': [], 'accuracy': []} for name in models}
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            continue
            
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            probs = model.predict_proba(X_val)[:, 1]
            
            results[name]['precision'].append(precision_score(y_val, preds, zero_division=0))
            results[name]['recall'].append(recall_score(y_val, preds, zero_division=0))
            results[name]['roc_auc'].append(roc_auc_score(y_val, probs))
            results[name]['accuracy'].append(accuracy_score(y_val, preds))
            
            # Save final fold for plotting
            if fold == 4:
                fpr, tpr, _ = roc_curve(y_val, probs)
                last_fold_roc[name] = (fpr, tpr, roc_auc_score(y_val, probs))
                last_fold_cm[name] = confusion_matrix(y_val, preds)
                
    # Plot ROC-AUC for all models
    plt.figure(figsize=(8, 6))
    for name, (fpr, tpr, auc_val) in last_fold_roc.items():
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc_val:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Validation ROC Curve Comparison (Time Series Split Fold 5)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(PLOTS_DIR, "validation_roc_curves.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    
    best_model_name = "XGBoost"
    best_prec = 0.0
    
    report_md = "# Quantitative Validation Report\n\n## TimeSeries Cross-Validation Results\n"
    for name in models.keys():
        avg_p = np.mean(results[name]['precision'])
        avg_r = np.mean(results[name]['recall'])
        avg_auc = np.mean(results[name]['roc_auc'])
        avg_acc = np.mean(results[name]['accuracy'])
        
        row = f"- **{name}**: Precision: {avg_p:.4f} | Recall: {avg_r:.4f} | Accuracy: {avg_acc:.4f} | ROC-AUC: {avg_auc:.4f}\n"
        report_md += row
        logger.info(row.strip())
        
        if avg_p > best_prec:
            best_prec = avg_p
            best_model_name = name
            
    logger.info(f"Selected {best_model_name} as best model.")
    report_md += f"\n**Best Model Selected (Optimizing Precision): {best_model_name}**\n\n"
    
    # Plot Confusion Matrix of best model
    plt.figure(figsize=(6, 5))
    sns.heatmap(last_fold_cm[best_model_name], annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted (0=Fade, 1=Squeeze)')
    plt.ylabel('True')
    plt.title(f'{best_model_name} Confusion Matrix (Validation Set)')
    plt.tight_layout()
    cm_path = os.path.join(PLOTS_DIR, "best_model_cm.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    
    # Feature Importance (if XGBoost or LightGBM)
    final_model = models[best_model_name]
    final_model.fit(X, y)
    
    if hasattr(final_model, 'feature_importances_'):
        importances = final_model.feature_importances_
        indices = np.argsort(importances)[-15:] # Top 15
        
        plt.figure(figsize=(10, 8))
        plt.title(f'{best_model_name} Top Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        feat_path = os.path.join(PLOTS_DIR, "real_feature_importance.png")
        plt.savefig(feat_path, dpi=150)
        plt.close()
    
    logger.info("6. Exporting Best Model to Production...")
    model_path = os.path.join(MODELS_DIR, "production_alpha_model.pkl")
    joblib.dump(final_model, model_path)
    joblib.dump(final_model, os.path.join(MODELS_DIR, "xgboost_alpha.pkl")) # Fallback legacy name
    logger.info(f"Successfully exported {best_model_name} to {model_path}")
    
    # Write report
    report_md += f"T-Test RVOL (Winners vs Losers): T-Stat={t_stat:.2f}, P-Value={p_val:.4f}\n"
    with open(os.path.join(ARTIFACT_DIR, "quant_validation_report.md"), "w") as f:
        f.write(report_md)
        
    logger.info("Quant Pipeline Execution Complete.")

if __name__ == "__main__":
    run_quant_pipeline()
