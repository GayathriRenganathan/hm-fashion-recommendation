import pandas as pd
import numpy as np
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
from typing import Dict, Callable, List

def train_als_model(
    train: pd.DataFrame,
    factors: int,
    iterations: int,
    regularization: float,
    alpha: float,
    random_seed: int
):
    """
    Prepare data and train an ALS model, returning the model and mappings.

    Args:
        train (pd.DataFrame): Training data with customer and article IDs.
        factors (int): Number of latent factors.
        iterations (int): Number of ALS training iterations.
        regularization (float): Regularization parameter.
        alpha (float): Confidence scaling factor for implicit feedback.
        random_seed (int): Random seed for reproducibility.

    Returns:
        als_model, user_map, item_map, reverse_item_map, item_user_matrix
    """

    user_ids = train['customer_id'].unique()
    item_ids = train['article_id'].unique()

    user_map = {id_: idx for idx, id_ in enumerate(user_ids)}
    item_map = {id_: idx for idx, id_ in enumerate(item_ids)}
    reverse_item_map = {idx: id_ for id_, idx in item_map.items()}

    print("1. User map shape:", len(user_map))
    print("2. Item map shape:", len(item_map))
    print("3. Reverse item map shape:", len(reverse_item_map))

    train = train.copy()
    train['user_idx'] = train['customer_id'].map(user_map)
    train['item_idx'] = train['article_id'].map(item_map)

    last_date = train['t_dat'].max()
    days_diff = (last_date - train['t_dat']).dt.days
    train["time_decay"] = 1.0 / (1 + days_diff)
    
    purchase_counts = train.groupby(['user_idx', 'item_idx'])["time_decay"].sum().reset_index(name='count')

    # User-Item sparse matrix
    user_item_matrix  = sp.coo_matrix(
        (purchase_counts['count'].astype(np.float32),
            (purchase_counts['user_idx'], purchase_counts['item_idx'])),
        shape=(len(user_map), len(item_map))
    ).tocsr()
    # Train ALS model
    als_model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=random_seed,
        alpha=alpha
    )
    
    print("Training Implicit ALS model...")
    als_model.fit(user_item_matrix, show_progress=True)
    print("Training complete!")
    
    return als_model, user_map, item_map, reverse_item_map, user_item_matrix


def create_als_recommender(
    train: pd.DataFrame = None,
    factors: int = 100,
    iterations: int = 15,
    regularization: float = 0.01,
    alpha: float = 40.0,
    random_seed: int = 42,
    baseline: Callable = None,
    als_model: AlternatingLeastSquares = None,
    user_map: Dict[str, int] = None,
    item_map: Dict[str, int] = None,
    reverse_item_map: Dict[int, str] = None,
    user_item_matrix: sp.csr_matrix = None
) -> Callable:
    """
    Create an ALS-based recommender function.

    Args:
        train (pd.DataFrame, optional): Training data for fitting ALS if model not provided.
        factors (int): Number of latent factors.
        iterations (int): Number of ALS training iterations.
        regularization (float): Regularization parameter.
        alpha (float): Confidence scaling factor.
        random_seed (int): Random seed for reproducibility.
        baseline (Callable, optional): Fallback recommender for cold start.
        als_model (AlternatingLeastSquares, optional): Pretrained ALS model.
        user_map, item_map, reverse_item_map, item_user_matrix: Precomputed mappings and matrix.

    Returns:
        Callable: Function that generates recommendations using ALS model.
    """
    if als_model is None:
        als_model, user_map, item_map, reverse_item_map, user_item_matrix = train_als_model(
            train,
            factors,
            iterations,
            regularization,
            alpha,
            random_seed
        )

    def als_recommender(customers: List[str], k: int = 12) -> pd.DataFrame:
        """
        Generate recommendations for a list of customers using ALS model.

        Args:
            customers (List[str]): List of customer IDs.
            k (int): Number of recommendations per customer.

        Returns:
            pd.DataFrame: DataFrame with columns [customer_id, rank, article_id].
        """
        recommendations = []
        customer_ids = []
        ranks = []

        for customer in customers:
            if customer not in user_map:
                if baseline is not None:
                    # Use baseline for cold start
                    recs = baseline([customer], k=k)['article_id'].tolist()
                else:
                    recs = []
            else:
                user_idx = user_map[customer]
                recommended = als_model.recommend(
                    user_idx,
                    user_item_matrix[user_idx],
                    N=k,
                    filter_already_liked_items=False
                )
                recs = [str(reverse_item_map[int(item_idx)]) for item_idx in recommended[0]]

            recommendations.extend(recs)
            customer_ids.extend([customer] * len(recs))
            ranks.extend(range(1, len(recs) + 1))

        return pd.DataFrame({
            "customer_id": customer_ids,
            "rank": ranks,
            "article_id": recommendations
        })

    return als_recommender
