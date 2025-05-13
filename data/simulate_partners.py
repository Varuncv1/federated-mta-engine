#!/usr/bin/env python3
"""
simulate_partners.py: Simulate multi-partner ad impression and conversion data for federated attribution modeling.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

np.random.seed(42)

PARTNERS = ["partner_a", "partner_b", "partner_c"]
N_USERS = 1000
IMPRESSIONS_PER_USER = (1, 5)
START_DATE = datetime(2024, 1, 1)


def generate_impressions(partner_name):
    data = []
    for user_id in range(N_USERS):
        n_imps = np.random.randint(*IMPRESSIONS_PER_USER)
        timestamp = START_DATE + timedelta(days=np.random.randint(0, 30))
        converted = int(np.random.rand() < 0.2)  # 20% conversion rate

        for i in range(n_imps):
            data.append({
                "user_id": f"user_{user_id}",
                "timestamp": timestamp + timedelta(minutes=i * 10),
                "channel": partner_name,
                "converted": converted
            })
    return pd.DataFrame(data)


def main():
    os.makedirs("data", exist_ok=True)
    for partner in PARTNERS:
        df = generate_impressions(partner)
        out_path = f"data/{partner}.csv"
        df.to_csv(out_path, index=False)
        print(f"Generated {len(df)} rows for {partner} at {out_path}")


if __name__ == "__main__":
    main()
