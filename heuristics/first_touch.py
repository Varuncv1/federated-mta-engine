# heuristics/first_touch.py
#!/usr/bin/env python3
"""
first_touch.py: Assigns 100% credit to the first touchpoint for each converted user.
"""

import pandas as pd

def first_touch_attribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns full credit to the first impression for each user who converted.

    Args:
        df: DataFrame with columns ['user_id', 'timestamp', 'channel', 'converted']

    Returns:
        DataFrame with columns ['user_id', 'channel', 'credit']
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_sorted = df.sort_values(['user_id', 'timestamp'])
    first = df_sorted.groupby('user_id').first().reset_index()
    result = first[['user_id', 'channel']].copy()
    result['credit'] = 1.0
    return result


if __name__ == "__main__":
    import os
    # Load all partner data
    files = [os.path.join('data', f) for f in os.listdir('data') if f.endswith('.csv')]
    df_all = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df_conv = df_all[df_all['converted'] == 1]

    ft = first_touch_attribution(df_conv)
    os.makedirs('data', exist_ok=True)
    ft.to_csv('data/first_touch_credit.csv', index=False)
    print('First-touch credits saved to data/first_touch_credit.csv')
