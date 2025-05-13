#!/usr/bin/env python3
"""
federated_trainer.py: Perform federated averaging of local models and evaluate the aggregated model.
"""
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from local_model import LocalModel, extract_features

PARTNERS = ["partner_a", "partner_b", "partner_c"]
LOCAL_MODEL_DIR = os.path.join("models", "local")


def load_local_state_dicts(partners):
    state_dicts = []
    for partner in partners:
        model_path = os.path.join(LOCAL_MODEL_DIR, f"{partner}_model.pth")
        state = torch.load(model_path, map_location="cpu")
        state_dicts.append(state)
    return state_dicts


def average_state_dicts(state_dicts):
    """Average the weights across multiple state_dicts."""
    avg_state = {}
    # Initialize avg_state with zeros
    for key in state_dicts[0].keys():
        avg_state[key] = torch.zeros_like(state_dicts[0][key])
    # Sum
    for state in state_dicts:
        for key, param in state.items():
            avg_state[key] += param
    # Average
    for key in avg_state:
        avg_state[key] = avg_state[key] / len(state_dicts)
    return avg_state


def evaluate_global_model(model, test_loader, device):
    model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    accuracy = correct / total * 100
    print(f"Global Model Test Accuracy: {accuracy:.2f}%")
    return accuracy


def main():
    # 1. Load local state dicts
    print("Loading local model state_dicts...")
    state_dicts = load_local_state_dicts(PARTNERS)

    # 2. Average weights
    print("Averaging state_dicts...")
    avg_state = average_state_dicts(state_dicts)

    # 3. Initialize global model and load averaged weights
    # Assume all local models use same architecture
    dummy_model = LocalModel(input_dim=3)
    dummy_model.load_state_dict(avg_state)

    # 4. Prepare aggregated test data
    print("Loading test data...")
    # Combine partner test sets
    dfs = []
    for partner in PARTNERS:
        df = pd.read_csv(os.path.join("data", f"{partner}.csv"))
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    features, labels = extract_features(df_all)

    # Scale features using global scaler fit on combined data
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)
    y = labels.values

    # Create DataLoader
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    test_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=64)

    # 5. Evaluate global model
    print("Evaluating global model...")
    evaluate_global_model(dummy_model, test_loader, device=torch.device("cpu"))

    # 6. Save global model
    os.makedirs("models/global", exist_ok=True)
    global_path = os.path.join("models", "global", "global_model.pth")
    torch.save(dummy_model.state_dict(), global_path)
    print(f"Global model saved to {global_path}")

if __name__ == "__main__":
    main()
