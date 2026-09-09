"""
Insider Trading Detection Model
Trained on historical SEC enforcement cases
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score
import pickle
import os

class InsiderTradingDetector:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = [
            'days_before_news',
            'volume_spike_ratio', 
            'price_correlation',
            'account_concentration'
        ]
        
    def load_training_data(self, filepath):
        """Load SEC enforcement cases"""
        df = pd.read_csv(filepath)
        return df
    
    def train(self, training_data_path, model_save_path='models/insider_trading_model.pkl'):
        """Train the model on historical SEC enforcement cases"""
        print("Loading training data from SEC enforcement cases...")
        df = self.load_training_data(training_data_path)
        
        # Prepare features and labels
        X = df[self.feature_names].values
        y = df['is_insider_trading'].values
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.25, random_state=42
        )
        
        # Train Random Forest
        print(f"Training Random Forest model on {len(X_train)} cases...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'  # Handle imbalanced data
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print("\n=== MODEL PERFORMANCE ===")
        print(f"Precision: {precision:.3f} (catches real violations)")
        print(f"Recall: {recall:.3f} (doesn't miss cases)")
        print(f"F1-Score: {f1:.3f}")
        print(f"AUC-ROC: {auc:.3f}")
        print(f"False Positive Rate: {(1-precision):.1%}")
        
        # Save model
        model_dir = os.path.dirname(model_save_path) or 'models'
        os.makedirs(model_dir, exist_ok=True)
        with open(model_save_path, 'wb') as f:
            pickle.dump((self.model, self.scaler), f)
        print(f"\nModel saved to {model_save_path}")
        
        return self
    
    def load_model(self, model_path):
        """Load pre-trained model"""
        with open(model_path, 'rb') as f:
            self.model, self.scaler = pickle.load(f)
        return self
    
    def predict_risk(self, trade_features):
        """
        Predict insider trading risk for a trade
        
        Args:
            trade_features: dict with keys:
                - days_before_news
                - volume_spike_ratio
                - price_correlation
                - account_concentration
        
        Returns:
            risk_score (0-100), probability of insider trading
        """
        features = np.array([
            [
                trade_features['days_before_news'],
                trade_features['volume_spike_ratio'],
                trade_features['price_correlation'],
                trade_features['account_concentration']
            ]
        ])
        
        features_scaled = self.scaler.transform(features)
        probability = self.model.predict_proba(features_scaled)[0, 1]
        risk_score = int(probability * 100)
        
        return risk_score, probability
    
    def get_feature_importance(self):
        """Get which features matter most"""
        importances = self.model.feature_importances_
        features_with_importance = list(zip(self.feature_names, importances))
        features_with_importance.sort(key=lambda x: x[1], reverse=True)
        return features_with_importance


if __name__ == "__main__":
    # Train model
    detector = InsiderTradingDetector()
    detector.train('data/sec_enforcement_cases.csv')
    
    # Test prediction
    test_trade = {
        'days_before_news': 2,
        'volume_spike_ratio': 8.5,
        'price_correlation': 0.92,
        'account_concentration': 0.78
    }
    
    risk_score, probability = detector.predict_risk(test_trade)
    print(f"\nTest Trade Risk Score: {risk_score}/100")
    print(f"Probability of Insider Trading: {probability:.1%}")
    
    # Feature importance
    print("\nFeature Importance:")
    for feature, importance in detector.get_feature_importance():
        print(f"  {feature}: {importance:.3f}")
