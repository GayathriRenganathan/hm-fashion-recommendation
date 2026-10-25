import pandas as pd
from typing import List
from ..models.baseline_model import create_recency_baseline  

class RecencyCandidates:
    def __init__(self, transactions: pd.DataFrame, articles: pd.DataFrame, lookback_weeks: int = 12, k: int = 20):
        self.transactions = transactions
        self.articles = articles
        self.lookback_weeks = lookback_weeks
        self.k = k
        self.recommender_func = None
        
        # Initial fit on full data
        self._update_recommender(transactions)


    def _update_recommender(self, history: pd.DataFrame) -> None:
        
        history = self.transactions[
            (self.transactions["week"] < self.lookback_weeks) 
        ].copy()
        
        if history.empty:
            self.recommender_func = lambda customers, k: pd.DataFrame(columns=["customer_id", "article_id", "rank"])
        else:
            self.recommender_func = create_recency_baseline(history)

    def generate(self, users: List[str]) -> pd.DataFrame:
        if len(users) == 0 or self.recommender_func is None:
            return pd.DataFrame(columns=["customer_id", "article_id", "rank"])
            
        df = self.recommender_func(users, k=self.k)
        # Drop the local rank column 
        if "rank" in df.columns:
            df = df.drop(columns=["rank"])
            
        return df