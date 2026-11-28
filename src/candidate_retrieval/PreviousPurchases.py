from src.candidate_retrieval.BaseCandidateGenerator import BaseCandidateGenerator

from typing import List
import pandas as pd

class PreviousPurchases(BaseCandidateGenerator):
    """
    Generates recommendation candidates from customers' previous purchases.

    Retrieves the most recently purchased unique articles within a
    configurable lookback window.

    For each customer, the generator returns up to K unique articles,
    ordered by purchase recency.
    """
    
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
        # Retrieves the most recent unique purchases for the specified users.
        filtered_history = self.get_recent_history(self.lookback_weeks)
        filtered_history = filtered_history[
            filtered_history['customer_id'].isin(users)
        ]
        
        # Sort by date (most recent first) 
        filtered_history = filtered_history[['customer_id', 'article_id', 't_dat']]\
            .sort_values(by=['customer_id', 't_dat'], ascending=[True, False])\
            .copy()
        # Drop duplicates based on customer and article,
        filtered_history = filtered_history.drop_duplicates(
            subset=['customer_id', 'article_id'], 
            keep='first'
        ).copy()
        # Take the top k most recent unique purchases for each user
        result = filtered_history[['customer_id', 'article_id']]\
            .groupby('customer_id')\
            .head(self.k)\
            .reset_index(drop=True)
        print("Generated previous purchase candidates")
        return result