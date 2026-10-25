from src.candidate_retrieval.BaseCandidateGenerator import BaseCandidateGenerator

from typing import List
import pandas as pd

class PreviousPurchases(BaseCandidateGenerator):
    def __init__(
        self, 
        transactions: pd.DataFrame, 
        articles: pd.DataFrame,
        lookback_weeks: int = 12,
        k: int = 12
    ):
        super().__init__(transactions, articles)
        self.lookback_weeks = lookback_weeks
        self.k = k

    def generate(self, users: List[str]) -> pd.DataFrame:
        # Filter transactions for the specified customers in your training set and 
        # filter transactions until specific lookback period (e.g., last 12 weeks)
        filtered_history = self.transactions[
            self.transactions['customer_id'].isin(users) & 
            (self.transactions['week'] <= self.lookback_weeks)
        ]
        
        # Sort by date (most recent first) and remove duplicate article purchases per user
        filtered_history = filtered_history[['customer_id', 'article_id', 't_dat']]\
            .sort_values(by='t_dat', ascending=False)\
            .copy()
        # Drop duplicates based on customer and article, keeping the 'first' (most recent)
        filtered_history = filtered_history.drop_duplicates(
            subset=['customer_id', 'article_id'], 
            keep='first'
        ).copy()
        # Take the top k most recent unique purchases for each user
        result = filtered_history[['customer_id', 'article_id']]\
            .groupby('customer_id')\
            .head(self.k)\
            .reset_index(drop=True)
        
        return result