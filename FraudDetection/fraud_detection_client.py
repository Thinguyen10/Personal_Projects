# Full implementation: Real-time fraud detection client
# - Loads a model (or uses a dummy fallback)
# - Connects to a TCP server and reads newline-delimited JSON transactions
# - Processes transactions and prints alerts/stats

import os
import json
import socket
from collections import deque
import random

try:
    import torch
except Exception:
    torch = None


class _DummyModel:
    """Fallback lightweight model exposing predict(features) -> float"""
    def predict(self, features):
        try:
            amt = float(features.get('amount', 0) if isinstance(features, dict) else features[0])
        except Exception:
            amt = 0.0
        prob = 1 / (1 + 2.71828 ** (-0.01 * (amt - 100)))
        return min(max(prob * (0.9 + 0.2 * random.random()), 0.0), 1.0)


class RealTimeFraudDetector:
    def __init__(self, model_path=None, host='localhost', port=5555):
        self.model = self.load_model(model_path)
        self.host = host
        self.port = port
        self.user_histories = {}
        self.detection_buffer = deque(maxlen=100)
        self.stats = {'total': 0, 'flagged': 0, 'true_positives': 0, 'false_positives': 0}
        self.running = False

    def load_model(self, model_path):
        if not model_path:
            return _DummyModel()
        if torch is not None and os.path.exists(model_path):
            try:
                m = torch.load(model_path, map_location='cpu')
                if hasattr(m, 'predict'):
                    return m

                class _Wrap:
                    def __init__(self, m):
                        self.m = m

                    def predict(self, features):
                        import numpy as _np
                        arr = _np.atleast_2d(list(features.values()) if isinstance(features, dict) else features)
                        with torch.no_grad():
                            t = torch.tensor(arr, dtype=torch.float32)
                            out = self.m(t)
                            try:
                                return float(out.detach().numpy().ravel()[0])
                            except Exception:
                                return float(out.ravel()[0])

                return _Wrap(m)
            except Exception:
                return _DummyModel()
        return _DummyModel()

    def extract_features(self, transaction):
        return {
            'amount': transaction.get('amount', 0.0),
            'hourOfDay': transaction.get('hourOfDay', 0),
            'isWeekend': int(transaction.get('isWeekend', False)),
            'daysSinceLastTransaction': transaction.get('daysSinceLastTransaction', 0)
        }

    def update_user_history(self, transaction):
        uid = transaction.get('userID')
        if uid is None:
            return
        hist = self.user_histories.setdefault(uid, deque(maxlen=20))
        hist.append(transaction.get('amount', 0.0))

    def update_stats(self, transaction, flagged):
        self.stats['total'] += 1
        if flagged:
            self.stats['flagged'] += 1
            if transaction.get('isFraud'):
                self.stats['true_positives'] += 1
            else:
                self.stats['false_positives'] += 1

    def alert_fraud(self, transaction, confidence):
        print("\n🚨 FRAUD ALERT 🚨")
        print(f"Transaction ID: {transaction.get('transactionID')}")
        print(f"User ID: {transaction.get('userID')}")
        try:
            amt = transaction.get('amount', 0.0)
            print(f"Amount: ${amt:.2f}")
        except Exception:
            print(f"Amount: {transaction.get('amount')}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Pattern: {transaction.get('fraudPattern', 'Unknown')}")

    def process_transaction(self, transaction):
        features = self.extract_features(transaction)
        fraud_prob = self.model.predict(features)
        self.update_user_history(transaction)
        is_fraud = fraud_prob > 0.5
        if is_fraud:
            self.alert_fraud(transaction, fraud_prob)
        self.update_stats(transaction, is_fraud)
        return fraud_prob

    def run(self, reconnect=False, timeout=10):
        self.running = True
        while self.running:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(timeout)
                    print(f"Connecting to {self.host}:{self.port}...")
                    s.connect((self.host, self.port))
                    print(f"Connected to {self.host}:{self.port}")

                    buffer = ''
                    while self.running:
                        try:
                            data = s.recv(4096).decode('utf-8')
                        except socket.timeout:
                            continue
                        if not data:
                            print('Connection closed by server')
                            break
                        buffer += data
                        lines = buffer.split('\n')
                        buffer = lines[-1]
                        for line in lines[:-1]:
                            if not line.strip():
                                continue
                            try:
                                txn = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            self.process_transaction(txn)

            except (ConnectionRefusedError, OSError) as e:
                print(f"Connection error: {e}")
                if not reconnect:
                    break
                print('Retrying in 2 seconds...')
                import time
                time.sleep(2)
                continue

            if not reconnect:
                break


# CLI
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Real-time fraud detection client')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', default=5555, type=int)
    parser.add_argument('--model', default=None, help='Path to model file (optional)')
    parser.add_argument('--reconnect', action='store_true', help='Keep retrying if connection lost')
    args = parser.parse_args()

    detector = RealTimeFraudDetector(model_path=args.model, host=args.host, port=args.port)
    try:
        detector.run(reconnect=args.reconnect)
    except KeyboardInterrupt:
        print('\nShutting down client...')
        detector.running = False