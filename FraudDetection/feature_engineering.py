import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime

class TransactionFeatures:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = [
            'amount', 'hour', 'isWeekend', 
            'daysSinceLastTransaction',
            'amount_zscore',  # Amount deviation from user's mean
            'time_since_midnight',  # Seconds since midnight
            'merchant_risk_score',  # Based on merchant ID pattern
            # TODO: Add more engineered features
        ]
    
    def transform_transaction(self, transaction, user_history=None):
        """Convert transaction to feature vector"""
        # TODO: Implement feature extraction
        # TODO: Include user history features if available
        pass
    def _merchant_risk(self, merchant_id: str) -> float:
        # Simple heuristic: numeric part parity + length mapping
        if not merchant_id:
            return 0.5
        digits = ''.join([c for c in merchant_id if c.isdigit()])
        if not digits:
            return 0.5
        val = int(digits) % 10
        return (val / 9.0)  # normalized 0..1

    def transform_transaction(self, transaction: dict, user_history: list | None = None):
        """Convert a single transaction dict into a numeric feature vector.

        Args:
            transaction: dict with keys amount, timestamp, merchantID, isWeekend, hourOfDay, daysSinceLastTransaction
            user_history: optional list of previous amounts for this user
        Returns:
            feature_vector (np.array), feature_names (list)
        """
        amount = float(transaction.get('amount', 0.0))
        hour = int(transaction.get('hourOfDay', datetime.fromisoformat(transaction['timestamp']).hour) if transaction.get('timestamp') else transaction.get('hourOfDay', 0))
        is_weekend = 1 if transaction.get('isWeekend', False) else 0
        days_since = float(transaction.get('daysSinceLastTransaction', 0.0))

        # amount_zscore based on user's history
        if user_history and len(user_history) >= 2:
            mu = float(np.mean(user_history))
            sigma = float(np.std(user_history)) if float(np.std(user_history)) > 1e-6 else 1.0
            amount_z = (amount - mu) / sigma
        else:
            amount_z = 0.0

        # seconds since midnight
        if transaction.get('timestamp'):
            try:
                t = datetime.fromisoformat(transaction['timestamp'])
                time_since_midnight = t.hour * 3600 + t.minute * 60 + t.second
            except Exception:
                time_since_midnight = hour * 3600
        else:
            time_since_midnight = hour * 3600

        merchant_risk = self._merchant_risk(transaction.get('merchantID', ''))

        features = [
            amount,
            hour,
            is_weekend,
            days_since,
            amount_z,
            time_since_midnight,
            merchant_risk
        ]

        return np.array(features, dtype=float), self.feature_names