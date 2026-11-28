from abc import ABC, abstractmethod
import pandas as pd
from typing import List

class BaseCandidateGenerator(ABC):
    def __init__(self, transactions: pd.DataFrame, articles: pd.DataFrame):
        self.transactions = transactions
        self.articles = articles
     
    
    def get_recent_history(self, lookback_weeks: int = 12) -> pd.DataFrame:
        """
        Returns transactions within the specified lookback window.

        Smaller week values represent more recent weeks.
        The latest available week is determined dynamically.
        """
        if self.transactions.empty:
            return self.transactions.copy()

        if lookback_weeks is None:
            return self.transactions.copy()

        latest_week = self.transactions['week'].min()
        end_week = latest_week + lookback_weeks - 1

        return self.transactions[
            self.transactions['week'].between(latest_week, end_week)
        ].copy()
        
    @abstractmethod
    def generate(self, users: List[str]) -> pd.DataFrame:
        pass