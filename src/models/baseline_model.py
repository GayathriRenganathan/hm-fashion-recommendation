"""
This module provides simple baseline recommenders for benchmarking, including global popularity,
temporal popularity, and random recommenders.

"""

import pandas as pd
from typing import Callable, List
import numpy as np

def create_baseline_recommender(train: pd.DataFrame) -> Callable:
    """
    Create a baseline recommender based on global popularity.
    
    Args:
        train: Training data DataFrame
        
    Returns:
        Function that generates recommendations based on global popularity
    """
    top = train.groupby("article_id")\
        .size()\
        .sort_values(ascending=False)\
        .reset_index(name="count")\
        .article_id\
        .tolist()
    
    def baseline(customers: List[str], k: int = 12) -> pd.DataFrame:
        """
        Generate top-k global popularity recommendations for a given list of customers.
        Args:
            customers: List of customer IDs
            k: Number of recommendations per customer
            
        Returns:
            pd.DataFrame with the following columns:
                - 'customer_id': The ID of the customer (repeated k times).
                - 'rank': The recommendation ranking position (ranging from 1 to k).
                - 'article_id': The globally popular article ID assigned to that rank.
        """
        return pd.DataFrame({
            "customer_id": [customer for customer in customers for _ in range(k)],
            "rank": list(range(1, k+1)) * len(customers),
            "article_id": top[:k] * len(customers)
        })
    
    return baseline