#!/usr/bin/env python3
import sys
import flwr as fl
import torch
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from models.local_model import LocalModel, extract_features
from opacus import PrivacyEngine

class MTAClient(fl.client.NumPyClient):
    def __init__(self, partner: str):
        # 1) Load model skeleton
        self.model = LocalModel(input_dim=3)

        # 2) Prepare local data
        scaler = joblib.load(f"models/local/{partner}_scaler.pkl")
        df = pd.read_csv(f"data/{partner}.csv")
        features, labels = extract_features(df)
        X = scaler.transform(features.values)
        self.train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(labels.values, dtype=torch.long),
            ),
            batch_size=32,
            shuffle=True,
        )

    def get_parameters(self):
        # Return model weights as NumPy arrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        # 1) Load incoming global parameters
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict)
        self.model.train()

        # 2) Attach Opacus PrivacyEngine for DP
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        privacy_engine = PrivacyEngine(
            self.model,
            batch_size=32,
            sample_size=len(self.train_loader.dataset),
            alphas=[10, 100],
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )
        privacy_engine.attach(optimizer)

        # 3) Local training
        for X_batch, y_batch in self.train_loader:
            optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = torch.nn.CrossEntropyLoss()(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # 4) (Optional) get current ε for reporting
        epsilon, best_alpha = privacy_engine.accountant.get_privacy_spent(delta=1e-5)
        print(f"[{self.__class__.__name__}] ε = {epsilon:.2f}, α = {best_alpha}")

        # 5) Return updated weights and local sample count
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        # Similar to server evaluate, but on local data if desired
        # Here we skip implementing it for simplicity:
        return 0.0, len(self.train_loader.dataset), {}

if __name__ == "__main__":
    partner = sys.argv[1]  # e.g. 'partner_a'
    fl.client.start_numpy_client(
        server_address="0.0.0.0:8080",
        client=MTAClient(partner),
    )
