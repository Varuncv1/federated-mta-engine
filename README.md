# Federated Multi-Touch Attribution Engine

This repository demonstrates a privacy-preserving, federated approach to multi-touch attribution (MTA) modeling—enabling multiple data partners to collaboratively train a shared attribution model without exposing raw user data.

## Project Structure

```
federated_mta/
├── data/                   # Simulated partner impression & conversion logs
│   ├── partner_a.csv
│   ├── partner_b.csv
│   └── partner_c.csv
│   └── simulate_partners.py
├── heuristics/             # Attribution heuristics (first-touch, linear)
│   ├── first_touch.py
│   └── linear.py
├── models/                 # Model definitions & training scripts
│   ├── local_model.py      # Train local partner models
│   ├── federated_trainer.py# Average local models into global model
│   └── global/             # Saved global model
│   └── local/              # Saved local models & scalers
├── evaluation/             # Evaluation metrics and analysis
│   ├── metrics.py          # Metric functions: AUC, log loss, correlation, etc.
│   └── notebooks/          # Jupyter notebooks for results
│       └── results_analysis.ipynb
├── notebooks/              # (Alternative location for analysis notebooks)
└── README.md               # This file
```

## Environment Setup

1. **Clone the repo**:

   ```bash
   git clone <repo_url>
   cd federated_mta
   ```
2. **Create & activate** a Python environment (e.g., Conda or venv):

   ```bash
   conda create -n mta python=3.9
   conda activate mta
   ```
3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   *Expected packages:* `pandas`, `numpy`, `torch`, `scikit-learn`, `joblib`, `flower` (optional), `opacus` (optional).

## Data Simulation

Generate synthetic impression & conversion logs for three partners:

```bash
python data/simulate_partners.py
```

Outputs: `data/partner_a.csv`, `partner_b.csv`, `partner_c.csv`.

## Attribution Heuristics

Compute per-user credit using:

* **First-Touch**:

  ```bash
  python heuristics/first_touch.py
  ```
* **Linear**:

  ```bash
  python heuristics/linear.py
  ```

Outputs: `data/first_touch_credit.csv` and `data/linear_credit.csv`.

## 🤖 Local Model Training

Train a conversion prediction model on each partner’s data:

```bash
python models/local_model.py --partner partner_a --epochs 10 --lr 0.01
python models/local_model.py --partner partner_b
python models/local_model.py --partner partner_c
```

Locally saved models: `models/local/{partner}_model.pth` and `{partner}_scaler.pkl`.

## Federated Learning Loop

Aggregate local models into a global model:

```bash
python models/federated_trainer.py
```

Saves `models/global/global_model.pth` and prints global test accuracy on combined data.

## Evaluation & Analysis

Open the Jupyter notebook to compute metrics and visualize results:

```
jupyter notebook evaluation/notebooks/results_analysis.ipynb
```

**Notebook Outline**:

1. **Imports & Setup**: Load dependencies and helper functions.
2. **Data & Models**: Read local/global model outputs and credit files.
3. **Metrics Calculation**: Use `evaluation/metrics.py` to compute AUC, log loss, and attribution correlation.
4. **Visualization**: Plot performance comparison (local vs. federated vs. centralized) and attribution quality.
5. **Privacy Trade-off**: (Optional) Analyze model accuracy under DP noise.

## Differential Privacy Experiments

To add privacy noise during federated updates, integrate **Opacus** in your training clients:

```python
from opacus import PrivacyEngine
# ... wrap your optimizer
privacy_engine = PrivacyEngine(model, batch_size=..., sample_size=..., alphas=[...], noise_multiplier=1.0, max_grad_norm=1.0)
optimizer, data_loader, model = privacy_engine.make_private(
    module=model, optimizer=optimizer, data_loader=train_loader
)
```

Analyze the privacy-utility trade-off in the evaluation notebook.

---

*Feel free to open issues or contribute enhancements!*
