#!/usr/bin/env python3
import flwr as fl
import torch
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from models.local_model import LocalModel, extract_features
from evaluation.metrics import compute_classification_metrics

# List of partners to aggregate data for evaluation
PARTNERS = ["partner_a", "partner_b", "partner_c"]

def get_parameters():
    # Initialize a fresh model to send as initial parameters
    model = LocalModel(input_dim=3)
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

class FedAvgWithEval(fl.server.strategy.FedAvg):
    def __init__(self, fraction_fit, fraction_eval, min_fit_clients,
                 min_eval_clients, num_rounds):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_eval=fraction_eval,
            min_fit_clients=min_fit_clients,
            min_eval_clients=min_eval_clients,
            initial_parameters=get_parameters()
        )
        self.num_rounds = num_rounds

    def evaluate(
        self,
        parameters,       # numpy weights from clients
        config            # config from latest round
    ):
        """Evaluate aggregated model on combined partner data."""
        # 1) Load averaged parameters into model
        model = LocalModel(input_dim=3)
        state_dict = {
            k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)
        }
        model.load_state_dict(state_dict)
        model.eval()

        # 2) Build combined test set
        dfs = []
        for p in PARTNERS:
            df = pd.read_csv(os.path.join("data", f"{p}.csv"))
            dfs.append(df)
        df_all = pd.concat(dfs, ignore_index=True)
        features, labels = extract_features(df_all)

        # 3) Scale features
        scaler = StandardScaler().fit(features.values)
        X = scaler.transform(features.values)
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # 4) Predict
        with torch.no_grad():
            outputs = model(X_tensor)
            probs = torch.softmax(outputs, 1)[:, 1].numpy()

        # 5) Compute metrics
        metrics = compute_classification_metrics(labels.values, probs)
        loss = metrics["log_loss"]
        # Return (loss, {"accuracy": ...})
        return loss, {"accuracy": metrics["accuracy"]}

if __name__ == "__main__":
    # Tune these as needed
    strategy = FedAvgWithEval(
        fraction_fit=0.8,
        fraction_eval=0.5,
        min_fit_clients=3,
        min_eval_clients=3,
        num_rounds=5,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=strategy.num_rounds),
        strategy=strategy,
    )
