from src.candidate_retrieval.BaseCandidateGenerator import BaseCandidateGenerator

import pandas as pd
from typing import Any, Dict, List, Tuple


class CandidateGeneratorEnsemble(BaseCandidateGenerator):
    """
    An ensemble candidate generator that combines multiple independent 
    candidate generators into a single unified pipeline.
    """
    
    def __init__(self,
        transactions: pd.DataFrame,
        articles: pd.DataFrame,
        generators: Dict[str, Tuple[BaseCandidateGenerator, Dict[str, Any]]]
    ):
        super().__init__(transactions, articles)
        
        # Instantiate each generator with its configuration parameters
        self.generators = {}
        for name, (generator_class, kwargs) in generators.items():
            self.generators[name] = generator_class(transactions, articles, **kwargs)

    def generate(self, users: List[str]) -> pd.DataFrame:
        """
        Executes all generators independently, merges their candidate pools,
        and deduplicates while preserving generator priority order.
        """
        all_recommendations = pd.DataFrame()

        # Run each generator independently
        for name, generator in self.generators.items():
            new_recs = generator.generate(users)
            if new_recs is not None and not new_recs.empty:
                all_recommendations = pd.concat([all_recommendations, new_recs])


        #Sort by customer_id
        all_recommendations = all_recommendations.sort_values(by="customer_id",ascending=False)

        # Deduplicate, keeping the first occurrence (respects generator registration order)
        return all_recommendations.drop_duplicates().reset_index(drop=True)