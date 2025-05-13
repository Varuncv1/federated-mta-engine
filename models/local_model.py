#!/usr/bin/env python3
"""
local_model.py: Train a local logistic regression model for a given partner's ad data.
"""
import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class LocalModel(nn.Module):
    def __init__(self, input_dim):
        super(LocalModel, self).__init__()
        # Simple logistic regression for conversion prediction
        self.fc = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.fc(x)


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw impressions into per-user features.
    Features:
      - impressions_count: number of touchpoints
      - hours_span: hours between first and last impression
      - first_hour: hour of day of first impression
    Label:
      - conversion (0 or 1)
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Group by user
    agg = df.groupby('user_id').agg(
        impressions_count=('timestamp', 'count'),
        first_ts=('timestamp', 'min'),
        last_ts=('timestamp', 'max'),
        conversion=('converted', 'max')
    ).reset_index()
    # Compute spans and first impression hour
    agg['hours_span'] = (agg['last_ts'] - agg['first_ts']).dt.total_seconds() / 3600.0
    agg['first_hour'] = agg['first_ts'].dt.hour
    # Select features
    features = agg[['impressions_count', 'hours_span', 'first_hour']]
    labels = agg['conversion'].astype(int)
    return features, labels


def train_local_model(partner: str, epochs: int, lr: float, batch_size: int, test_size: float):
    # Load partner data
    data_path = os.path.join('data', f'{partner}.csv')
    df = pd.read_csv(data_path)

    # Feature extraction
    X_df, y = extract_features(df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_df.values, y.values, test_size=test_size, random_state=42, stratify=y.values
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # DataLoaders
    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Model, loss, optimizer
    input_dim = X_train.shape[1]
    model = LocalModel(input_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Partner: {partner} | Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    accuracy = correct / total * 100
    print(f"Partner: {partner} | Test Accuracy: {accuracy:.2f}%")

    # Save model state and scaler
    os.makedirs('models/local', exist_ok=True)
    model_path = os.path.join('models', 'local', f'{partner}_model.pth')
    scaler_path = os.path.join('models', 'local', f'{partner}_scaler.pkl')
    torch.save(model.state_dict(), model_path)
    # Save scaler
    import joblib
    joblib.dump(scaler, scaler_path)
    print(f"Saved model to {model_path} and scaler to {scaler_path}")

    return model.state_dict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train local model for a partner.")
    parser.add_argument('--partner', type=str, required=True, help='Partner name (e.g., partner_a)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_size', type=float, default=0.2)
    args = parser.parse_args()

    train_local_model(
        partner=args.partner,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        test_size=args.test_size
    )
