# heuristics/linear.py
#!/usr/bin/env python3
"""
linear.py: Distribute equal credit across all touchpoints for each converted user.
"""

import pandas as pd

def linear_attribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Splits credit equally among all impressions for each user who converted.

    Args:
        df: DataFrame with columns ['user_id', 'timestamp', 'channel', 'converted']

    Returns:
        DataFrame with columns ['user_id', 'channel', 'credit']
    """
    df = df.copy()
    counts = df.groupby('user_id').size().rename('touch_count')
    df = df.join(counts, on='user_id')
    df['credit'] = 1.0 / df['touch_count']
    return df[['user_id', 'channel', 'credit']]


if __name__ == "__main__":
    import os
    # Load all partner data
    files = [os.path.join('data', f) for f in os.listdir('data') if f.endswith('.csv')]
    df_all = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df_conv = df_all[df_all['converted'] == 1]

    lin = linear_attribution(df_conv)
    os.makedirs('data', exist_ok=True)
    lin.to_csv('data/linear_credit.csv', index=False)
    print('Linear credits saved to data/linear_credit.csv')
