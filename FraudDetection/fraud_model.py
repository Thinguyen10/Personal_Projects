import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class FraudDetectionNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=None, dropout=0.3):
        super(FraudDetectionNet, self).__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            in_dim = h

        # Final output layer -> single probability
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x).squeeze(-1)


class FraudDetector:
    def __init__(self, input_dim=10, model_path=None, lr=1e-3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = FraudDetectionNet(input_dim=input_dim).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        if model_path:
            self.load_model(model_path)

    def save_model(self, path: str):
        """Save model state_dict to the given path."""
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        """Load model state_dict from the given path into the current model."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        state = torch.load(path, map_location=self.device)
        # If the saved object is a state_dict, load it
        if isinstance(state, dict):
            self.model.load_state_dict(state)
        else:
            # Otherwise assume a full model object; attempt to set state_dict
            try:
                self.model.load_state_dict(state.state_dict())
            except Exception:
                # As fallback, try to set model to loaded object
                self.model = state.to(self.device)

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64, patience=5, plot=True):
        """Train the fraud detection model.

        Args:
            X_train, y_train, X_val, y_val: numpy arrays or torch tensors
            epochs: max epochs
            batch_size: training batch size
            patience: early stopping patience on validation loss
            plot: if True and matplotlib available, save loss plot to training_plot.png

        Returns:
            history dict with train_loss and val_loss lists
        """
        # Convert inputs to tensors if needed
        if not torch.is_tensor(X_train):
            X_train_t = torch.tensor(X_train, dtype=torch.float32)
        else:
            X_train_t = X_train.float()

        if not torch.is_tensor(y_train):
            y_train_t = torch.tensor(y_train, dtype=torch.float32)
        else:
            y_train_t = y_train.float()

        if not torch.is_tensor(X_val):
            X_val_t = torch.tensor(X_val, dtype=torch.float32)
        else:
            X_val_t = X_val.float()

        if not torch.is_tensor(y_val):
            y_val_t = torch.tensor(y_val, dtype=torch.float32)
        else:
            y_val_t = y_val.float()

        train_ds = TensorDataset(X_train_t, y_train_t)
        val_ds = TensorDataset(X_val_t, y_val_t)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        best_val_loss = float('inf')
        best_state = None
        epochs_no_improve = 0
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(1, epochs + 1):
            # Training
            self.model.train()
            running_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(xb)
                loss = self.criterion(preds, yb)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * xb.size(0)

            train_loss = running_loss / len(train_loader.dataset)

            # Validation
            self.model.eval()
            val_loss_accum = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    preds = self.model(xb)
                    loss = self.criterion(preds, yb)
                    val_loss_accum += loss.item() * xb.size(0)
                    # compute accuracy
                    predicted = (preds >= 0.5).float()
                    correct += (predicted == yb).sum().item()
                    total += xb.size(0)

            val_loss = val_loss_accum / len(val_loader.dataset)
            val_acc = correct / total if total else 0.0

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            print(f"Epoch {epoch:03d}/{epochs} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")

            # Early stopping
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered (no improvement in {patience} epochs)")
                    break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Optional plotting
        if plot:
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(8, 4))
                plt.plot(history['train_loss'], label='train_loss')
                plt.plot(history['val_loss'], label='val_loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.tight_layout()
                plt.savefig('training_plot.png')
                plt.close()
                print('Saved training plot to training_plot.png')
            except Exception:
                # plotting is optional; ignore errors
                pass

        return history

    def predict(self, transaction_features):
        """Predict fraud probability for a single transaction feature vector.

        transaction_features: array-like or torch tensor of shape (input_dim,)
        returns: float probability in [0,1]
        """
        self.model.eval()
        if not torch.is_tensor(transaction_features):
            x = torch.tensor(transaction_features, dtype=torch.float32)
        else:
            x = transaction_features.float()

        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = x.to(self.device)
        with torch.no_grad():
            out = self.model(x)
            # out shape (batch,), take first
            prob = float(out.squeeze(0).cpu().item())
        return prob