from abc import ABC, abstractmethod
import pandas as pd
from typing import List

class BaseCandidateGenerator(ABC):
    def __init__(self, transactions: pd.DataFrame, articles: pd.DataFrame):
        self.transactions = transactions
        self.articles = articles
     
    @abstractmethod
    def generate(self, users: List[str]) -> pd.DataFrame:
        pass