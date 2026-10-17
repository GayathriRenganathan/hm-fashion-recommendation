"""
This module provides simple baseline recommenders for benchmarking, including global popularity,
recent popularity.

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

def create_recency_baseline(train: pd.DataFrame) -> Callable:
    """
    Create a temporal baseline recommender based on recent popularity.
    
    Args:
        train (pd.DataFrame): Training data DataFrame containing 't_dat' and 'article_id'.
        
    Returns:
        Callable: Function that generates recommendations based on recent popularity.
    """
    train = train.copy()
    train['t_dat'] = pd.to_datetime(train['t_dat'])
    last_date = train['t_dat'].max()
    
    days_diff = (last_date - train['t_dat']).dt.days
    train["time_decay"] = 1 / (1 + days_diff)

    # Compute sum of time-decayed weights per article
    top_temporal = (
        train.groupby("article_id")["time_decay"]
        .sum()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    
    def recency_recommender(customers: List[str], k: int = 12) -> pd.DataFrame:
        """
        Generate recommendations for a list of customers using recent popularity.
        
        Args:
            customers (List[str]): List of customer IDs.
            k (int): Number of recommendations per customer.
            
        Returns:
            pd.DataFrame: Recommendation table with customer_id, rank, and article_id.
        """
        return pd.DataFrame({
            "customer_id": [customer for customer in customers for _ in range(k)],
            "rank": list(range(1, k+1)) * len(customers),
            "article_id": top_temporal[:k] * len(customers)
        })
    
    return recency_recommender