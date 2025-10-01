import json
import socket
import pandas as pd
from datetime import datetime
import pickle

class FraudDataCollector:
    def __init__(self, host='localhost', port=5555, collection_size=1000):
        """
        Args:
            host: 'localhost' for your own server
                  or IP address like '192.168.1.10' for shared server
            port: Usually 5555
            collection_size: Number of transactions to collect
        """
        self.host = host
        self.port = port
        self.collection_size = collection_size
        self.transactions = []
    
    def connect_and_collect(self):
        """Connect to server and collect transactions"""
        # Reset transactions buffer
        self.transactions = []

        # Create a TCP/IP socket and connect to the server.
        # We expect the server to send one JSON transaction per-line (newline-delimited).
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                # Short timeout to avoid hanging indefinitely on connect/read
                s.settimeout(10)
                s.connect((self.host, self.port))

                # Wrap socket in a file-like object so we can read line-by-line
                # This is easier and more robust for newline-delimited JSON streams.
                with s.makefile('r') as sf:
                    while len(self.transactions) < self.collection_size:
                        line = sf.readline()
                        # If the server closed the connection, readline() returns ''
                        if not line:
                            break
                        line = line.strip()
                        if not line:
                            # skip empty lines
                            continue
                        try:
                            txn = json.loads(line)
                        except json.JSONDecodeError:
                            # Skip malformed JSON lines but keep reading
                            continue
                        # Append the parsed transaction (expected to be a dict)
                        self.transactions.append(txn)
        except (socket.timeout, ConnectionRefusedError, OSError) as e:
            # If connection fails or is interrupted, return None so caller can handle it.
            # In a real app you might want to log this or retry.
            return None

        # After collection, save raw transactions to CSV with a timestamped filename.
        if self.transactions:
            df = pd.DataFrame(self.transactions)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"transactions_{timestamp}.csv"
            # Save DataFrame to CSV (no index). This will contain the raw fields sent by server.
            df.to_csv(filename, index=False)
            return filename

        # If no transactions were collected, return None
        return None
    
    def extract_features(self, transaction):
        """Extract features for ML model"""
        # Basic raw features
        features = {
            'amount': transaction.get('amount', 0.0),
            'hour': transaction.get('hourOfDay', 0),
            'isWeekend': int(transaction.get('isWeekend', False)),
            'daysSinceLastTransaction': transaction.get('daysSinceLastTransaction', 0),
        }

        # Engineered features (simple heuristics)
        # amount_zscore and time_since_midnight require user history or timestamp;
        # we provide sensible defaults here; downstream code can replace with
        # more accurate engineered values if user history is available.
        features.update({
            'amount_zscore': 0.0,  # placeholder; compute with user history when available
            'time_since_midnight': 0.0,
            'merchant_risk_score': 0.5
        })
        return features