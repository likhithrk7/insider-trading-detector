"""
BACKTESTING & PERFORMANCE VISUALIZATION
Comprehensive analysis of ensemble model performance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve, roc_curve, confusion_matrix,
    classification_report, roc_auc_score, auc
)
import pickle
import json

class BacktestingEngine:
    """Comprehensive backtesting and performance analysis"""
    
    def __init__(self, model_path='models/ensemble_model.pkl'):
        with open(model_path, 'rb') as f:
            models_dict = pickle.load(f)
        
        self.rf_model = models_dict['rf']
        self.xgb_model = models_dict['xgb']
        self.nn_model = models_dict['nn']
        self.scaler = models_dict['scaler']
        self.feature_names = models_dict.get('feature_names', [])
    
    def engineer_features(self, df, feature_cols):
        """Create engineered features"""
        X = df[feature_cols].copy()
        X['days_before_news_squared'] = X['days_before_news'] ** 2
        X['volume_price_interaction'] = X['volume_spike_ratio'] * X['price_correlation']
        X['concentration_momentum'] = X['account_concentration'] * X['volatility_zscore']
        X['risk_intensity'] = (
            X['volume_spike_ratio'] * X['price_correlation'] * X['account_concentration']
        )
        return X
    
    def backtest(self, training_data_path='data/sec_enforcement_cases_expanded.csv'):
        """Run comprehensive backtest"""
        print("="*70)
        print("BACKTESTING ENGINE - COMPREHENSIVE PERFORMANCE ANALYSIS")
        print("="*70)
        
        # Load data
        print("\n[1/6] Loading historical enforcement data...")
        df = pd.read_csv(training_data_path)
        
        # Engineering features - use only the features model was trained on
        base_features = [
            'days_before_news', 'volume_spike_ratio', 'price_correlation',
            'holding_period_days', 'account_concentration', 'volatility_zscore',
            'options_activity', 'sector_correlation', 'account_history_violations'
        ]
        available_features = [f for f in base_features if f in df.columns]
        X = self.engineer_features(df, available_features)
        y = df['is_insider_trading'].values
        
        X_scaled = self.scaler.transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.25, random_state=42, stratify=y
        )
        
        print(f"      Training samples: {len(X_train)}")
        print(f"      Test samples: {len(X_test)}")
        
        # Get predictions
        print("\n[2/6] Generating model predictions...")
        rf_pred_test = self.rf_model.predict_proba(X_test)[:, 1]
        xgb_pred_test = self.xgb_model.predict_proba(X_test)[:, 1]
        nn_pred_test = self.nn_model.predict_proba(X_test)[:, 1]
        
        ensemble_pred = (rf_pred_test + xgb_pred_test + nn_pred_test) / 3
        ensemble_binary = (ensemble_pred > 0.5).astype(int)
        
        # Calculate metrics
        print("\n[3/6] Calculating performance metrics...")
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, ensemble_binary, average='binary'
        )
        auc_score = roc_auc_score(y_test, ensemble_pred)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, ensemble_binary).ravel()
        
        # Detailed metrics
        true_positive_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        print("\n" + "="*70)
        print("BACKTEST RESULTS")
        print("="*70)
        
        print("\nCONFUSION MATRIX:")
        print(f"  True Negatives:  {tn:4d}  |  False Positives: {fp:4d}")
        print(f"  False Negatives: {fn:4d}  |  True Positives:  {tp:4d}")
        
        print("\nPERFORMANCE METRICS:")
        print(f"  Precision:  {precision:.4f} (of predicted violations, {precision:.1%} correct)")
        print(f"  Recall:     {recall:.4f} (of actual violations, {recall:.1%} caught)")
        print(f"  F1-Score:   {f1:.4f}")
        print(f"  AUC-ROC:    {auc_score:.4f}")
        print(f"  True Positive Rate:  {true_positive_rate:.1%}")
        print(f"  False Positive Rate: {false_positive_rate:.1%}")
        print(f"  False Negative Rate: {false_negative_rate:.1%}")
        
        # Model comparison
        print("\nMODEL COMPARISON (Test Set Performance):")
        print("  " + "-"*60)
        
        rf_auc = roc_auc_score(y_test, rf_pred_test)
        xgb_auc = roc_auc_score(y_test, xgb_pred_test)
        nn_auc = roc_auc_score(y_test, nn_pred_test)
        
        print(f"  Random Forest:  AUC = {rf_auc:.4f}")
        print(f"  XGBoost:        AUC = {xgb_auc:.4f}")
        print(f"  Neural Network: AUC = {nn_auc:.4f}")
        print(f"  Ensemble (Avg): AUC = {auc_score:.4f}")
        
        # Risk distribution
        print("\n[4/6] Analyzing risk score distribution...")
        risk_scores = (ensemble_pred * 100).astype(int)
        
        print("\nRISK SCORE DISTRIBUTION:")
        print(f"  Critical (80+):  {(risk_scores >= 80).sum():3d} cases")
        print(f"  High (60-79):    {((risk_scores >= 60) & (risk_scores < 80)).sum():3d} cases")
        print(f"  Medium (40-59):  {((risk_scores >= 40) & (risk_scores < 60)).sum():3d} cases")
        print(f"  Low (<40):       {(risk_scores < 40).sum():3d} cases")
        
        # Precision at different thresholds
        print("\n[5/6] Evaluating performance at different risk thresholds...")
        print("\nPRECISION AT DIFFERENT THRESHOLDS:")
        print("  Threshold | Alerts | Precision | Recall")
        print("  " + "-"*45)
        
        for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
            alerts = (ensemble_pred >= threshold).sum()
            if alerts > 0:
                threshold_tp = ((ensemble_pred >= threshold) & (y_test == 1)).sum()
                threshold_precision = threshold_tp / alerts if alerts > 0 else 0
                threshold_recall = threshold_tp / (y_test == 1).sum() if (y_test == 1).sum() > 0 else 0
                print(f"    {threshold:.1f}    |   {alerts:3d}  |   {threshold_precision:.1%}    | {threshold_recall:.1%}")
        
        # Generate metrics report
        print("\n[6/6] Generating detailed report...")
        
        metrics = {
            'dataset': {
                'total_samples': len(X_test),
                'positive_cases': int((y_test == 1).sum()),
                'negative_cases': int((y_test == 0).sum()),
                'class_balance': float((y_test == 1).sum() / len(y_test))
            },
            'ensemble_performance': {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'auc_roc': float(auc_score),
                'true_positive_rate': float(true_positive_rate),
                'false_positive_rate': float(false_positive_rate),
                'false_negative_rate': float(false_negative_rate)
            },
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            },
            'individual_models': {
                'random_forest_auc': float(rf_auc),
                'xgboost_auc': float(xgb_auc),
                'neural_network_auc': float(nn_auc)
            },
            'risk_distribution': {
                'critical_80plus': int((risk_scores >= 80).sum()),
                'high_60to79': int(((risk_scores >= 60) & (risk_scores < 80)).sum()),
                'medium_40to59': int(((risk_scores >= 40) & (risk_scores < 60)).sum()),
                'low_below40': int((risk_scores < 40).sum())
            }
        }
        
        # Save report
        with open('results/backtest_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print("\nMetrics saved to results/backtest_metrics.json")
        print("\n" + "="*70)
        print("BACKTEST COMPLETE")
        print("="*70)
        
        return metrics, ensemble_pred, y_test
    
    def generate_alerts_report(self, data_path='data/sample_trades.csv'):
        """Generate alerts report on real trades"""
        print("\n" + "="*70)
        print("TRADE ANALYSIS & ALERTS REPORT")
        print("="*70)
        
        # Load real trades
        trades_df = pd.read_csv(data_path)
        
        # Required base features (in correct order for scaler)
        base_feature_order = [
            'days_before_news', 'volume_spike_ratio', 'price_correlation',
            'holding_period_days', 'account_concentration', 'volatility_zscore',
            'options_activity', 'sector_correlation', 'account_history_violations'
        ]
        
        alerts = []
        print(f"\nAnalyzing {len(trades_df)} trades...")
        
        for idx, trade in trades_df.iterrows():
            # Create feature dict with proper values
            features_base = {
                'days_before_news': min(trade.get('days_to_news', 30), 30),
                'volume_spike_ratio': max(1.0, trade.get('volume_spike_ratio', 1.0)),
                'price_correlation': max(0.0, min(1.0, trade.get('price_correlation', 0.5))),
                'holding_period_days': trade.get('holding_period_days', 30),
                'account_concentration': max(0.0, min(1.0, trade.get('account_concentration', 0.5))),
                'volatility_zscore': 1.5,
                'options_activity': 0.6,
                'sector_correlation': 0.7,
                'account_history_violations': 0
            }
            
            # Create DataFrame to maintain feature order
            feature_df = pd.DataFrame([features_base])
            
            # Add engineered features (same as training)
            feature_df['days_before_news_squared'] = feature_df['days_before_news'] ** 2
            feature_df['volume_price_interaction'] = feature_df['volume_spike_ratio'] * feature_df['price_correlation']
            feature_df['concentration_momentum'] = feature_df['account_concentration'] * feature_df['volatility_zscore']
            feature_df['risk_intensity'] = feature_df['volume_spike_ratio'] * feature_df['price_correlation'] * feature_df['account_concentration']
            
            # Predict using properly ordered features
            feature_scaled = self.scaler.transform(feature_df)
            
            rf_prob = self.rf_model.predict_proba(feature_scaled)[0, 1]
            xgb_prob = self.xgb_model.predict_proba(feature_scaled)[0, 1]
            nn_prob = self.nn_model.predict_proba(feature_scaled)[0, 1]
            
            ensemble_prob = (rf_prob + xgb_prob + nn_prob) / 3
            risk_score = int(ensemble_prob * 100)
            
            if risk_score >= 50:  # Flag medium and above
                alerts.append({
                    'trade_id': trade.get('trade_id', f'TRADE_{idx}'),
                    'symbol': trade.get('symbol', 'N/A'),
                    'risk_score': risk_score,
                    'probability': ensemble_prob,
                    'days_to_news': features_base['days_before_news'],
                    'volume_spike': features_base['volume_spike_ratio'],
                    'price_correlation': features_base['price_correlation'],
                    'account_concentration': features_base['account_concentration']
                })
        
        if alerts:
            alerts_df = pd.DataFrame(alerts).sort_values('risk_score', ascending=False)
            alerts_df.to_csv('results/ensemble_alerts.csv', index=False)
            
            print(f"\nAlerts Generated: {len(alerts)}")
            print(f"  Critical (80+): {(alerts_df['risk_score'] >= 80).sum()}")
            print(f"  High (60-79): {((alerts_df['risk_score'] >= 60) & (alerts_df['risk_score'] < 80)).sum()}")
            print(f"  Medium (50-59): {((alerts_df['risk_score'] >= 50) & (alerts_df['risk_score'] < 60)).sum()}")
            
            print("\nTop Alerts:")
            for idx, alert in alerts_df.head(5).iterrows():
                print(f"  {alert['trade_id']:20s} | Risk: {alert['risk_score']:3d}/100")
            
            print(f"\nAlerts saved to results/ensemble_alerts.csv")


if __name__ == "__main__":
    engine = BacktestingEngine()
    metrics, predictions, y_test = engine.backtest()
    engine.generate_alerts_report()
