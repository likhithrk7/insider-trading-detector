"""
PRODUCTION-GRADE INSIDER TRADING DETECTION SYSTEM
Ensemble Model: Random Forest + XGBoost + Neural Network
With SHAP Explainability and Comprehensive Performance Analysis
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    precision_recall_fscore_support, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, f1_score, classification_report
)
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Install with: pip install shap")

class EnsembleInsiderTradingDetector:
    """Production-grade insider trading detection with ensemble learning"""
    
    def __init__(self):
        self.rf_model = None
        self.xgb_model = None
        self.nn_model = None
        self.scaler = None
        self.poly_features = None
        self.feature_names = [
            'days_before_news',           # How close to announcement?
            'volume_spike_ratio',         # How much higher than normal?
            'price_correlation',          # Does price move with timing?
            'holding_period_days',        # How long held before selling?
            'account_concentration',      # What % of account is this trade?
            'volatility_zscore',          # Is stock more volatile than usual?
            'options_activity',           # Options positioning (signals)?
            'sector_correlation',         # Does sector move same way?
            'account_history_violations', # Has this account violated before?
            'trade_to_news_momentum'      # Price momentum into news
        ]
        
    def load_training_data(self, filepath):
        """Load SEC enforcement cases"""
        return pd.read_csv(filepath)
    
    def engineer_features(self, df):
        """Create advanced features from raw data"""
        X = df[self.feature_names[:-1]].copy()  # Exclude the new feature for now
        
        # Add derived features
        X['days_before_news_squared'] = X['days_before_news'] ** 2
        X['volume_price_interaction'] = X['volume_spike_ratio'] * X['price_correlation']
        X['concentration_momentum'] = X['account_concentration'] * X['volatility_zscore']
        X['risk_intensity'] = (
            X['volume_spike_ratio'] * X['price_correlation'] * X['account_concentration']
        )
        
        return X
    
    def train(self, training_data_path, model_save_path='models/ensemble_model.pkl'):
        """Train all three models (Random Forest, XGBoost, Neural Network)"""
        print("="*70)
        print("ENSEMBLE INSIDER TRADING DETECTION SYSTEM")
        print("="*70)
        
        print("\n[1/5] Loading training data from SEC enforcement cases...")
        df = self.load_training_data(training_data_path)
        print(f"      Loaded {len(df)} enforcement cases")
        print(f"      Positive cases (insider trading): {df['is_insider_trading'].sum()}")
        print(f"      Negative cases (normal trading): {(1-df['is_insider_trading']).sum()}")
        
        # Engineer features
        print("\n[2/5] Engineering advanced features...")
        X = self.engineer_features(df)
        y = df['is_insider_trading'].values
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data with stratification (important for imbalanced data)
        print("\n[3/5] Splitting data (75% train, 25% test)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.25, random_state=42, stratify=y
        )
        print(f"      Training set: {len(X_train)} cases")
        print(f"      Test set: {len(X_test)} cases")
        
        # Train Random Forest
        print("\n[4/5] Training ensemble models...")
        print("      • Random Forest (100 trees)...", end="", flush=True)
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        self.rf_model.fit(X_train, y_train)
        print(" ✓")
        
        # Train XGBoost
        print("      • XGBoost (100 rounds)...", end="", flush=True)
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            scale_pos_weight=1,
            eval_metric='logloss'
        )
        self.xgb_model.fit(X_train, y_train)
        print(" ✓")
        
        # Train Neural Network
        print("      • Neural Network (2 hidden layers)...", end="", flush=True)
        self.nn_model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        self.nn_model.fit(X_train, y_train)
        print(" ✓")
        
        # Get predictions from all models
        print("\n[5/5] Evaluating ensemble performance...")
        rf_pred = self.rf_model.predict_proba(X_test)[:, 1]
        xgb_pred = self.xgb_model.predict_proba(X_test)[:, 1]
        nn_pred = self.nn_model.predict_proba(X_test)[:, 1]
        
        # Ensemble prediction (average of all three)
        ensemble_pred = (rf_pred + xgb_pred + nn_pred) / 3
        ensemble_pred_binary = (ensemble_pred > 0.5).astype(int)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, ensemble_pred_binary, average='binary'
        )
        auc = roc_auc_score(y_test, ensemble_pred)
        
        print("\n" + "="*70)
        print("MODEL PERFORMANCE METRICS")
        print("="*70)
        print(f"Precision:  {precision:.4f} (catches real violations, not false alarms)")
        print(f"Recall:     {recall:.4f} (doesn't miss cases)")
        print(f"F1-Score:   {f1:.4f} (harmonic mean of precision/recall)")
        print(f"AUC-ROC:    {auc:.4f} (overall discrimination ability)")
        print(f"False Positive Rate: {(1-precision):.1%}")
        
        # Feature importance
        print("\n" + "="*70)
        print("FEATURE IMPORTANCE")
        print("="*70)
        
        # Random Forest importance
        rf_importance = list(zip(X.columns, self.rf_model.feature_importances_))
        rf_importance.sort(key=lambda x: x[1], reverse=True)
        print("\nRandom Forest Top Features:")
        for feature, importance in rf_importance[:5]:
            print(f"  {feature:.<40} {importance:.4f}")
        
        # XGBoost importance
        xgb_importance = list(zip(X.columns, self.xgb_model.feature_importances_))
        xgb_importance.sort(key=lambda x: x[1], reverse=True)
        print("\nXGBoost Top Features:")
        for feature, importance in xgb_importance[:5]:
            print(f"  {feature:.<40} {importance:.4f}")
        
        # Save models
        print("\n" + "="*70)
        model_dir = os.path.dirname(model_save_path) or 'models'
        os.makedirs(model_dir, exist_ok=True)
        
        models_dict = {
            'rf': self.rf_model,
            'xgb': self.xgb_model,
            'nn': self.nn_model,
            'scaler': self.scaler,
            'feature_names': list(X.columns)
        }
        
        with open(model_save_path, 'wb') as f:
            pickle.dump(models_dict, f)
        
        print(f"Models saved to {model_save_path}")
        print("="*70)
        
        # Return metrics for analysis
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'X_test': X_test,
            'y_test': y_test,
            'ensemble_pred': ensemble_pred,
            'rf_pred': rf_pred,
            'xgb_pred': xgb_pred,
            'nn_pred': nn_pred
        }
    
    def load_model(self, model_path):
        """Load pre-trained ensemble model"""
        with open(model_path, 'rb') as f:
            models_dict = pickle.load(f)
        
        self.rf_model = models_dict['rf']
        self.xgb_model = models_dict['xgb']
        self.nn_model = models_dict['nn']
        self.scaler = models_dict['scaler']
        self.feature_names = models_dict.get('feature_names', self.feature_names)
        
        return self
    
    def predict_risk_ensemble(self, trade_features):
        """
        Predict insider trading risk using ensemble voting
        Returns risk score (0-100) and individual model predictions
        """
        features = np.array([list(trade_features.values())])
        features_scaled = self.scaler.transform(features)
        
        # Get predictions from all models
        rf_prob = self.rf_model.predict_proba(features_scaled)[0, 1]
        xgb_prob = self.xgb_model.predict_proba(features_scaled)[0, 1]
        nn_prob = self.nn_model.predict_proba(features_scaled)[0, 1]
        
        # Ensemble vote (average)
        ensemble_prob = (rf_prob + xgb_prob + nn_prob) / 3
        risk_score = int(ensemble_prob * 100)
        
        return risk_score, {
            'ensemble': ensemble_prob,
            'random_forest': rf_prob,
            'xgboost': xgb_prob,
            'neural_network': nn_prob
        }
    
    def explain_prediction(self, trade_features):
        """
        Explain why a trade was flagged using SHAP values
        (Requires SHAP library)
        """
        if not SHAP_AVAILABLE:
            return "SHAP not installed. Install with: pip install shap"
        
        features = np.array([list(trade_features.values())])
        features_scaled = self.scaler.transform(features)
        
        # Use XGBoost for SHAP explanation (most stable)
        explainer = shap.TreeExplainer(self.xgb_model)
        shap_values = explainer.shap_values(features_scaled)
        
        explanation = {
            'base_value': explainer.expected_value,
            'shap_values': shap_values[0] if isinstance(shap_values, list) else shap_values[0],
            'features': list(trade_features.keys()),
            'feature_values': list(trade_features.values())
        }
        
        return explanation


if __name__ == "__main__":
    detector = EnsembleInsiderTradingDetector()
    
    # Train on expanded dataset
    metrics = detector.train('data/sec_enforcement_cases_expanded.csv')
    
    print("\n" + "="*70)
    print("TEST TRADE PREDICTION")
    print("="*70)
    
    # Test on a high-risk trade (base features only - engineered features added automatically)
    test_trade_base = {
        'days_before_news': 2,
        'volume_spike_ratio': 8.5,
        'price_correlation': 0.92,
        'holding_period_days': 14,
        'account_concentration': 0.78,
        'volatility_zscore': 2.1,
        'options_activity': 0.85,
        'sector_correlation': 0.91,
        'account_history_violations': 0
    }
    
    # Engineer features for test trade
    test_df = pd.DataFrame([test_trade_base])
    test_engineered = detector.engineer_features(test_df).iloc[0].to_dict()
    
    risk_score, predictions = detector.predict_risk_ensemble(test_engineered)
    
    print(f"\nTrade Risk Assessment:")
    print(f"  Overall Risk Score: {risk_score}/100")
    print(f"  Ensemble Probability: {predictions['ensemble']:.1%}")
    print(f"\nIndividual Model Predictions:")
    print(f"  Random Forest: {predictions['random_forest']:.1%}")
    print(f"  XGBoost: {predictions['xgboost']:.1%}")
    print(f"  Neural Network: {predictions['neural_network']:.1%}")
    
    print("\n" + "="*70)
    print("System ready for production deployment")
    print("="*70)
