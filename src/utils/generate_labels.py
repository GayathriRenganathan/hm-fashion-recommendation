import pandas as pd

def create_labels(candidates_df, target_transactions_df):
    """
    Creates binary labels for the candidate pool using a pre-filtered target transaction set.
    
    Parameters:
    - candidates_df: pd.DataFrame containing at least ['customer_id', 'article_id']
    - target_transactions_df: pd.DataFrame containing the actual transactions for the target week
    
    Returns:
    - pd.DataFrame: candidates_df with a new 'target' column (1 if bought, 0 otherwise)
    """
    print("Extracting unique purchase pairs from target set...")
    
    # Extract unique customer-article pairs from the actual target week
    actual_purchases = target_transactions_df[['customer_id', 'article_id']].drop_duplicates()
    actual_purchases['target'] = 1
    
    print(f"Found {len(actual_purchases):,} unique customer-article purchase pairs in target week.")
    print(f"Total candidates before labeling: {len(candidates_df):,}")

    # Left join candidates with actual target purchases
    labeled_df = candidates_df.merge(
        actual_purchases, 
        on=['customer_id', 'article_id'], 
        how='left'
    )
    
    # Unmatched candidates get 0 (negatives), matched get 1 (positives)
    labeled_df['target'] = labeled_df['target'].fillna(0).astype(int)
    
    positive_count = (labeled_df['target'] == 1).sum()
    coverage_pct = (positive_count / len(labeled_df)) * 100
    
    print(f"Positive samples captured in candidate pool: {positive_count:,} ({coverage_pct:.2f}%)")
    
    return labeled_df