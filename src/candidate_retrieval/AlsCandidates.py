import os
import pickle
import pandas as pd
from typing import Callable, Optional
from candidate_retrieval.BaseCandidateGenerator import BaseCandidateGenerator
from src.models.als_model import create_als_recommender
from pathlib import Path

class ALSCandidates(BaseCandidateGenerator):
    def __init__(
        self, 
        transactions: pd.DataFrame,
        articles: pd.DataFrame,
        model_path: Path,
        k: int = 40,
        baseline: Optional[Callable] = None
    ):
        super().__init__(transactions, articles)
        self.k = k
        if not model_path.exists():
            raise FileNotFoundError(
                f"ALS model not found: {model_path}. "
                "Train the model before generating candidates."
            )
        with open(model_path, "rb") as f:
            model_artifact = pickle.load(f)

        self.recommender_func = create_als_recommender(
            als_model=model_artifact["model"],
            user_map=model_artifact["user_map"],
            item_map=model_artifact["item_map"],
            reverse_item_map=model_artifact["reverse_item_map"],
            user_item_matrix=model_artifact["user_item_matrix"],
            baseline=baseline
        )

    def generate(self, users: list) -> pd.DataFrame:
        if len(users) == 0 or self.recommender_func is None:
            return pd.DataFrame(columns=["customer_id", "article_id"])
            
        df = self.recommender_func(list(users), k=self.k)
        if "rank" in df.columns:
            df = df.drop(columns=["rank"])
        print("Generated als candidates")
        return df