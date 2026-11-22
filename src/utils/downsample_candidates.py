import pandas as pd
import numpy as np

def downsample_negatives(df, target_col='target', max_negatives_per_user=30, random_state=42):
    """
    Downsamples negative candidates while preserving all positive candidates.
    Caps negatives per customer to a maximum limit for memory efficiency.
    """
    print(f"Original dataset shape: {df.shape}")
    print(f"Original positive count: {(df[target_col] == 1).sum():,}")
    
    positives = df[df[target_col] == 1]
    negatives = df[df[target_col] == 0]
    
    print(f"Sampling up to {max_negatives_per_user} negatives per user...")
    sampled_negatives = (
        negatives.groupby('customer_id', group_keys=False)
        .apply(
            lambda x: x.sample(
                n=min(len(x), max_negatives_per_user),
                random_state=random_state
            )
        )
    )
    
    downsampled_df = pd.concat([positives, sampled_negatives], ignore_index=True)
    
    downsampled_df = (
        downsampled_df
        .sample(frac=1.0, random_state=random_state)
        .sort_values('customer_id')
        .reset_index(drop=True)
    )
    
    print(f"Downsampled dataset shape: {downsampled_df.shape}")
    new_pos_prop = downsampled_df[target_col].mean()
    print(f"New positive proportion: {new_pos_prop:.4f} ({new_pos_prop * 100:.2f}%)")
    
    return downsampled_df
