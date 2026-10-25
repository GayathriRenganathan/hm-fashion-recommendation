"""
Implementation of ranking evaluation metrics (MAP@K, HitRate, NDCG@K, Recall@K).

This module provides functions to compute ranking metrics 
for evaluating recommender system performance.
"""
import numpy as np
import pandas as pd
from typing import Callable, Dict, Tuple


def _prepare_eval_data(preds_df: pd.DataFrame, test: pd.DataFrame, k: int) -> Tuple[np.ndarray, dict, np.ndarray]:
    """Helper to prepare and pivot data ONCE for all metrics."""
    customers = test['customer_id'].unique()
    
    # Ground truth mapping
    actuals = test.groupby('customer_id')['article_id'].apply(set).to_dict()
    
    # Prediction matrix aligned to customers
    pred_matrix = (
        preds_df.pivot(index='customer_id', columns='rank', values='article_id')
        .reindex(customers)
        .values[:, :k]  # Ensure it strictly cuts off at k
    )
    
    return customers, actuals, pred_matrix

def mapk(pred_matrix: np.ndarray, actuals: dict, customers: np.ndarray, k: int = 12) -> float:
    """
    Calculate mean average precision at k (MAPK).
    
    Args:
        pred_matrix (np.ndarray): Matrix of predicted articles for each customer.
        actuals (dict): Dictionary mapping customer IDs to sets of actual articles.
        customers (np.ndarray): Array of customer IDs.
        k (int): Number of recommendations to consider.
        
    Returns:
        float: Mean average precision score at k.
    """
    
    
    actuals_list = [actuals.get(cust_id, set()) for cust_id in customers]
    
    hit_matrix = np.zeros_like(pred_matrix, dtype=np.float32)
    for i, row in enumerate(pred_matrix):
        hit_matrix[i] = [item in actuals_list[i] if pd.notna(item) else False for item in row]
    
    cumsum_hits = np.cumsum(hit_matrix, axis=1)
    inv_ranks = 1 / np.arange(1, k + 1)
    precisions = hit_matrix * (cumsum_hits * inv_ranks)
    
    actuals_length = np.fromiter(
        (len(a) if a else 1 for a in actuals_list),
        dtype=np.int32,
        count=len(actuals_list)
    )
    
    ap_per_customer = precisions.sum(axis=1) / np.minimum(k, actuals_length)
    
    return float(np.mean(ap_per_customer))

def hit_rate(pred_matrix: np.ndarray, actuals: dict, customers: np.ndarray) -> float:
    """Calculate Hit Rate at k (fraction of users with at least one correct recommendation)."""
    
    hits = 0
    total = len(customers)
    
    for i, cust_id in enumerate(customers):
        actual = actuals.get(cust_id, set())
        pred_row = pred_matrix[i]
        if any(p in actual for p in pred_row if pd.notna(p)):
            hits += 1
            
    return float(hits / total) if total > 0 else 0.0


def ndcg_at_k(pred_matrix: np.ndarray, actuals: dict, customers: np.ndarray, k: int = 12) -> float:
    """Calculate Normalized Discounted Cumulative Gain at k (NDCG@k)."""

    ndcg_scores = []
    inv_log_ranks = 1.0 / np.log2(np.arange(2, k + 2))
    
    for i, cust_id in enumerate(customers):
        actual = actuals.get(cust_id, set())
        if not actual:
            continue
            
        pred_row = pred_matrix[i]
        
        # DCG calculation
        dcg = 0.0
        for rank_idx, item in enumerate(pred_row[:k]):
            if pd.notna(item) and item in actual:
                dcg += inv_log_ranks[rank_idx]
                
        # IDCG calculation
        ideal_hits = min(len(actual), k)
        idcg = sum(inv_log_ranks[:ideal_hits])
        
        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
        else:
            ndcg_scores.append(0.0)
            
    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0

def recall_at_k(pred_matrix: np.ndarray, actuals: dict, customers: np.ndarray, k: int = 12) -> float:
    """
    Calculate Recall at k (fraction of true items captured in the top k recommendations).
    
    Args:
        pred_matrix (np.ndarray): Matrix of predicted articles for each customer.
        actuals (dict): Dictionary mapping customer IDs to sets of actual articles.
        customers (np.ndarray): Array of customer IDs.
        k (int): Number of recommendations to consider.
        
    Returns:
        float: Mean recall score at k.
    """
    recall_scores = []
    
    for i, cust_id in enumerate(customers):
        actual = actuals.get(cust_id, set())
        if not actual:
            continue  # Skip users with no ground truth purchases in the validation window
            
        pred_row = pred_matrix[i][:k]
        
        # Count how many actual items were successfully predicted in the top k
        hits = sum(1 for item in pred_row if pd.notna(item) and item in actual)
        
        # Recall = (relevant items recommended) / (total actual relevant items)
        recall = hits / len(actual)
        recall_scores.append(recall)
        
    return float(np.mean(recall_scores)) if recall_scores else 0.0

def evaluate_all(recommender: Callable, test: pd.DataFrame, k: int = 12) -> Dict[str, float]:
    """
    Generates predictions, prepares data, and evaluates all metrics.
    """
    customers_test = test['customer_id'].unique()
    
    print(f"Generating predictions for {len(customers_test):,} validation customers...")
    preds_df = recommender(customers_test)
    
    print("Preparing evaluation matrices...")
    customers, actuals, pred_matrix = _prepare_eval_data(preds_df, test, k)
    
    print("Computing metrics...")
    return {
        f"MAP@{k}": mapk(pred_matrix, actuals, customers, k),
        f"HitRate@{k}": hit_rate(pred_matrix, actuals, customers),
        f"NDCG@{k}": ndcg_at_k(pred_matrix, actuals, customers, k),
        f"Recall@{k}": recall_at_k(pred_matrix, actuals, customers, k)
    }
    
    
def evaluate_candidates(preds_df, test_df, k=100):   
    
    # 2. Add sequential rank so your preparation script doesn't throw a KeyError
    if not preds_df.empty and 'rank' not in preds_df.columns:
        preds_df['rank'] = preds_df.groupby('customer_id').cumcount() + 1
        
    print("Preparing evaluation matrices...")
    customers, actuals, pred_matrix = _prepare_eval_data(preds_df, test_df, k)
    
    # 3. Compute Recall@K using your existing function
    score = recall_at_k(pred_matrix, actuals, customers, k)
    print(f"Recall@{k}: {score:.4f}")
    return score