"""
Ranker model module for H&M Fashion Recommendations.
"""

import pandas as pd
import numpy as np
from lightgbm import LGBMRanker
from typing import List


def train_ranker(
        X: pd.DataFrame,
        y: pd.Series,
        train_baskets: np.ndarray,
        boosting_type: str,
        n_estimators: int,
        min_child_samples: int,
        learning_rate: float
    ):
    """
    Train a LightGBM ranker (LGBMRanker) using the provided features and labels.
    """
    drop_cols = ["customer_id", "article_id", "target"]
    
    # Keep only actual model features
    X = X.drop(columns=drop_cols, errors="ignore").copy()
    
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in categorical_cols:
        X[col] = X[col].astype('category')

    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type=boosting_type,
        n_estimators=n_estimators,
        importance_type='gain',
        min_child_samples=min_child_samples,
        learning_rate=learning_rate,
        verbose=-1
    )

    model.fit(X, y, group=train_baskets, categorical_feature=categorical_cols)

    return model, categorical_cols


def create_lgbm_recommender(
    ranker: LGBMRanker, 
    candidate_features: pd.DataFrame, 
    categorical_cols: List[str],
    train_feature_categories: dict
    ):
    """
    Create a LightGBM-based recommender function that scores and ranks candidate recommendations.
    """

    def lgbm_recommender(customers_list: List[str], k: int = 12) -> pd.DataFrame:
        """
        Generate ranked recommendations for a list of customers using the trained LGBM ranker.
        """
        recommendations = candidate_features.copy()
        # Ensure categorical columns match training categories
        for col in categorical_cols:
            if col in recommendations.columns and col in train_feature_categories:
                recommendations[col] = pd.Categorical(
                    recommendations[col],
                    categories=train_feature_categories[col]
                )
                
        drop_cols = [c for c in ["customer_id", "article_id", "target"] if c in recommendations.columns]
        
        recommendations = recommendations[recommendations['customer_id'].isin(customers_list)].copy()
        if recommendations.empty:
            return pd.DataFrame(columns=["customer_id", "article_id", "rank"])
        
        inference_features = recommendations.drop(columns=drop_cols, errors="ignore")
      
        unique_customers = recommendations['customer_id'].unique()
        batch_size = 5000
        scores_list = []
        for i in range(0, len(unique_customers), batch_size):
            batch_custs = unique_customers[i : i + batch_size]
            
            mask = recommendations['customer_id'].isin(batch_custs)
            batch_X = inference_features[mask]
            
            batch_scores = ranker.predict(batch_X)
            scores_list.append(pd.Series(batch_scores, index=batch_X.index))
                        
        recommendations["scores"] = pd.concat(scores_list)
        recommendations['rank'] = recommendations.groupby('customer_id')['scores']\
            .rank(method='first', ascending=False).astype(int)
  
        return recommendations[["customer_id", "article_id", "rank"]]\
            .sort_values(["customer_id", "rank"])\
            .groupby("customer_id")\
            .head(k)
    
    return lgbm_recommender