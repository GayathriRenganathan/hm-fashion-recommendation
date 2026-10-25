import os
import pickle
import pandas as pd
from typing import List, Callable, Optional
from src.models.als_model import create_als_recommender, train_als_model
from config import ROOT_DIR
from pathlib import Path

class ALSCandidates:
    def __init__(
        self, 
        transactions: pd.DataFrame, 
        articles: pd.DataFrame, 
        lookback_weeks: int = 12, 
        k: int = 40,
        factors: int = 100,
        iterations: int = 15,
        regularization: float = 0.01,
        alpha: float = 40.0,
        random_seed: int = 42,
        baseline: Optional[Callable] = None
    ):
        self.transactions = transactions
        self.articles = articles
        self.lookback_weeks = lookback_weeks
        self.k = k
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.alpha = alpha
        self.random_seed = random_seed
        self.baseline = baseline
        self.recommender_func = None
        
        script_dir = Path(__file__).resolve().parent
        self.cache_dir = (script_dir / "../../saved_models").resolve()
        # Initial fit on full provided data
        self._update_recommender(transactions)

    def _update_recommender(self, history: pd.DataFrame) -> None:
        history = self.transactions[
            (self.transactions["week"] < self.lookback_weeks) 
        ].copy()
        if history.empty:
            self.recommender_func = lambda customers, k: pd.DataFrame(columns=["customer_id", "article_id", "rank"])
            return
        
        os.makedirs(self.cache_dir, exist_ok=True)
        # Construct a unique path based on configuration hyperparameters
        als_file_path = self.cache_dir / f"als_model_w{self.lookback_weeks}_f{self.factors}.pkl"
        als_old_file_path = self.cache_dir / f"als_recommender_model.pkl"
        
        print(f"ALS model cache path: {als_old_file_path}")
        # 1. Try loading from cache
        if os.path.exists(als_old_file_path):
            print(f"Loading cached ALS model from {als_old_file_path}...")
            with open(als_old_file_path, 'rb') as f:
                model_artifact = pickle.load(f)
            
            als_model = model_artifact['model']
            user_map = model_artifact['user_map']
            item_map = model_artifact['item_map']
            reverse_item_map = model_artifact['reverse_item_map']
            user_item_matrix = model_artifact['user_item_matrix']
            print("ALS model successfully loaded from cache!")
        else:
            print("Training ALS model from scratch...")
            als_model, user_map, item_map, reverse_item_map, user_item_matrix = train_als_model(
                train=history,
                factors=self.factors,
                iterations=self.iterations,
                regularization=self.regularization,
                alpha=self.alpha,
                random_seed=self.random_seed
            )

            # 3. Save the ALS model and mappings using your exact dictionary format
            model_artifact = {
                'model': als_model,                    
                'user_map': user_map,               
                'item_map': item_map,               
                'reverse_item_map': reverse_item_map,
                'user_item_matrix': user_item_matrix, 
            }

            with open(als_file_path, 'wb') as f:
                pickle.dump(model_artifact, f)

            print("ALS model successfully saved!")
            
        self.recommender_func = create_als_recommender(
            train=history,
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
            alpha=self.alpha,
            random_seed=self.random_seed,
            baseline=self.baseline,
            als_model=als_model,
            user_map=user_map,
            item_map=item_map,
            reverse_item_map=reverse_item_map,
            user_item_matrix=user_item_matrix
        )

    def generate(self, users: list) -> pd.DataFrame:
        if len(users) == 0 or self.recommender_func is None:
            return pd.DataFrame(columns=["customer_id", "article_id"])
            
        df = self.recommender_func(list(users), k=self.k)
        if "rank" in df.columns:
            df = df.drop(columns=["rank"])
            
        return df