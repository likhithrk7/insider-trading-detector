"""
Real-time trade analysis for insider trading patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from insider_trading_model import InsiderTradingDetector
import json

class TradeAnalyzer:
    def __init__(self, model_path='models/insider_trading_model.pkl'):
        self.detector = InsiderTradingDetector()
        self.detector.load_model(model_path)
        self.alerts = []
        
    def calculate_volume_spike(self, current_volume, average_volume):
        """Calculate how unusual the trading volume is"""
        if average_volume == 0:
            return 1.0
        return current_volume / average_volume
    
    def calculate_price_correlation(self, days_to_news, volume_spike_ratio):
        """
        Estimate correlation between trade timing and news impact
        Real implementation would use actual price movements
        """
        # Closer to news + higher volume = higher correlation
        time_factor = 1.0 - (days_to_news / 30.0)  # Closer to news = higher
        volume_factor = min(volume_spike_ratio / 10.0, 1.0)  # Higher volume = higher
        correlation = (time_factor * 0.6 + volume_factor * 0.4)
        return min(correlation, 0.99)
    
    def calculate_account_concentration(self, trade_amount, account_portfolio_size=10000000):
        """What % of account is this trade?"""
        concentration = trade_amount / account_portfolio_size
        return min(concentration, 1.0)
    
    def analyze_trade(self, trade_row, baseline_volume=50000):
        """Analyze single trade for insider trading risk"""
        
        # Calculate features
        days_to_news = trade_row['days_to_news']
        volume_spike = self.calculate_volume_spike(
            trade_row['quantity'], 
            baseline_volume
        )
        price_correlation = self.calculate_price_correlation(
            days_to_news, 
            volume_spike
        )
        account_concentration = self.calculate_account_concentration(
            trade_row['volume_sold_usd']
        )
        
        # Create feature dict
        trade_features = {
            'days_before_news': days_to_news,
            'volume_spike_ratio': volume_spike,
            'price_correlation': price_correlation,
            'account_concentration': account_concentration
        }
        
        # Get risk score from model
        risk_score, probability = self.detector.predict_risk(trade_features)
        
        return {
            'trade_id': trade_row['trade_id'],
            'account_id': trade_row['account_id'],
            'symbol': trade_row['symbol'],
            'trade_date': trade_row['trade_date'],
            'quantity': trade_row['quantity'],
            'price': trade_row['price'],
            'amount_usd': trade_row['volume_sold_usd'],
            'days_to_news': days_to_news,
            'news_event': trade_row['news_event_type'],
            'volume_spike_ratio': volume_spike,
            'price_correlation': price_correlation,
            'account_concentration': account_concentration,
            'risk_score': risk_score,
            'probability': probability,
            'risk_level': self._classify_risk(risk_score)
        }
    
    def _classify_risk(self, score):
        """Classify risk level"""
        if score >= 80:
            return 'CRITICAL'
        elif score >= 60:
            return 'HIGH'
        elif score >= 40:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def analyze_trades_file(self, trades_file, output_file='results/alerts.json'):
        """Analyze all trades in file"""
        print("Loading trades...")
        trades_df = pd.read_csv(trades_file)
        
        results = []
        print(f"Analyzing {len(trades_df)} trades...\n")
        
        for idx, row in trades_df.iterrows():
            result = self.analyze_trade(row)
            results.append(result)
            
            # Print high-risk alerts
            if result['risk_score'] >= 70:
                print(f"🚨 ALERT - Trade {result['trade_id']}")
                print(f"   Account: {result['account_id']}")
                print(f"   Symbol: {result['symbol']}")
                print(f"   Trade Date: {result['trade_date']}")
                print(f"   Risk Score: {result['risk_score']}/100")
                print(f"   Volume Spike: {result['volume_spike_ratio']:.1f}x normal")
                print(f"   News Event: {result['news_event']}")
                print(f"   Days to News: {result['days_to_news']}")
                print()
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by risk score
        results_df_sorted = results_df.sort_values('risk_score', ascending=False)
        
        # Summary statistics
        print("\n=== ANALYSIS SUMMARY ===")
        print(f"Total Trades Analyzed: {len(results_df)}")
        print(f"Critical Risk (80+): {len(results_df[results_df['risk_score'] >= 80])}")
        print(f"High Risk (60-79): {len(results_df[results_df['risk_score'].between(60, 79)])}")
        print(f"Medium Risk (40-59): {len(results_df[results_df['risk_score'].between(40, 59)])}")
        print(f"Low Risk (<40): {len(results_df[results_df['risk_score'] < 40])}")
        
        print(f"\nAverage Risk Score: {results_df['risk_score'].mean():.1f}/100")
        print(f"Median Risk Score: {results_df['risk_score'].median():.1f}/100")
        
        # Save results
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        results_df_sorted.to_csv(output_file.replace('.json', '.csv'), index=False)
        print(f"\nResults saved to {output_file.replace('.json', '.csv')}")
        
        return results_df_sorted


if __name__ == "__main__":
    analyzer = TradeAnalyzer()
    results = analyzer.analyze_trades_file('data/sample_trades.csv')
