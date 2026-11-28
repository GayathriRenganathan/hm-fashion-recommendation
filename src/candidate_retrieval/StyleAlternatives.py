from src.candidate_retrieval.BaseCandidateGenerator import BaseCandidateGenerator
from typing import List
import pandas as pd

class SameProductColorCandidates(BaseCandidateGenerator):
    """
    Generates style alternative candidates based on customers' historical
    product type and color preferences.

    The generator uses two candidate retrieval tiers:

    1. Exact match:
       Identifies the customer's top 3 product type and color combinations
       based on purchase frequency and retrieves popular articles matching
       those combinations.

    2. Product type fallback:
       Identifies the customer's top 3 product types based on purchase
       frequency and retrieves popular articles within those product types,
       regardless of color.

    Article pools are precomputed using global purchase popularity within
    the specified lookback period. This avoids repeatedly sorting articles
    during candidate generation.

    Candidates from both tiers are combined, deduplicated,  and filtered to
    exclude articles already purchased by the customer. The final candidate
    set is limited to the configured number of candidates per customer.

    Customers without eligible transaction history or valid product type
    information receive an empty candidate set. 
    """
    
    def __init__(
        self, 
        transactions: pd.DataFrame, 
        articles: pd.DataFrame,
        lookback_weeks: int = 12,
        k: int = 12,
        pool_depth: int = 50
    ):
        super().__init__(transactions, articles)
        self.lookback_weeks = lookback_weeks
        self.k = k
        self.pool_depth = pool_depth
        
        # Attribute lookups
        self.article_pt = self.articles.set_index('article_id')['product_type_no'].to_dict()
        self.article_color = self.articles.set_index('article_id')['perceived_colour_master_id'].to_dict()
        
        # Global article popularity
        history_tx = self.transactions[self.transactions['week'] <= self.lookback_weeks]
        global_counts = history_tx['article_id'].value_counts()
        
        article_summary = self.articles[['article_id', 'product_type_no', 'perceived_colour_master_id']].copy()
        article_summary['popularity'] = article_summary['article_id'].map(global_counts).fillna(0)
        
        # Precompute Tier 1: Exact (Product Type + Color) Pool
        exact_summary = article_summary.dropna(
            subset=['product_type_no', 'perceived_colour_master_id']
        )

        exact_summary = exact_summary.sort_values(
            ['product_type_no', 'perceived_colour_master_id', 'popularity'],
            ascending=[True, True, False]
        )
        self.pool_exact = (
            exact_summary.groupby(['product_type_no', 'perceived_colour_master_id'], sort=False)
            .head(self.pool_depth)[['product_type_no', 'perceived_colour_master_id', 'article_id']]
        )
        
        # Precompute Tier 2: Product Type Only Pool (Broader fallback)
        pt_summary = article_summary.dropna(subset=['product_type_no'])
        pt_summary = pt_summary.sort_values(['product_type_no', 'popularity'], ascending=[True, False])
        self.pool_pt_only = (
            pt_summary.groupby('product_type_no', sort=False)
            .head(self.pool_depth)[['product_type_no', 'article_id']]
        )

    def generate(self, users: List[str]) -> pd.DataFrame:
        # Filter transactions to the lookback window and for the target users
        filtered_history = self.get_recent_history(self.lookback_weeks)
        user_tx = filtered_history[
            filtered_history['customer_id'].isin(users) 
        ]

        if user_tx.empty:
            return pd.DataFrame(columns=['customer_id', 'article_id'])

        batch_tx = user_tx[['customer_id', 'article_id']].copy()
        batch_tx['product_type_no'] = batch_tx['article_id'].map(self.article_pt)
        batch_tx['perceived_colour_master_id'] = batch_tx['article_id'].map(self.article_color)
        batch_tx = batch_tx.dropna(subset=['product_type_no'])

        if batch_tx.empty:
            return pd.DataFrame(columns=['customer_id', 'article_id'])

        # Tier 1: Exact Product Type + Color Preferences
        valid_exact = batch_tx.dropna(subset=['perceived_colour_master_id'])
        user_prefs_exact = (
            valid_exact.groupby(['customer_id', 'product_type_no', 'perceived_colour_master_id'], sort=False)
            .size().reset_index(name='count')
        )
        user_prefs_exact['rank'] = user_prefs_exact.groupby('customer_id', sort=False)['count'].rank(method='first', ascending=False)
        top_exact = user_prefs_exact[user_prefs_exact['rank'] <= 3][['customer_id', 'product_type_no', 'perceived_colour_master_id']]

        exact_candidates = top_exact.merge(self.pool_exact, on=['product_type_no', 'perceived_colour_master_id'], how='inner')
        exact_candidates = exact_candidates[['customer_id', 'article_id']].drop_duplicates()

        # Tier 2: Product Type Only Preferences (Relaxed color constraint)
        user_prefs_pt = (
            batch_tx.groupby(['customer_id', 'product_type_no'], sort=False)
            .size().reset_index(name='count')
        )
        user_prefs_pt['rank'] = user_prefs_pt.groupby('customer_id', sort=False)['count'].rank(method='first', ascending=False)
        top_pt = user_prefs_pt[user_prefs_pt['rank'] <= 3][['customer_id', 'product_type_no']]

        pt_candidates = top_pt.merge(self.pool_pt_only, on=['product_type_no'], how='inner')
        pt_candidates = pt_candidates[['customer_id', 'article_id']].drop_duplicates()

        # Combine tiers and ensure top K per user
        combined = pd.concat([exact_candidates, pt_candidates], ignore_index=True).drop_duplicates()
        
        # Exclude already purchased articles
        purchased = user_tx[['customer_id', 'article_id']].drop_duplicates()

        combined = combined.merge(
            purchased.assign(purchased=True),
            on=['customer_id', 'article_id'],
            how='left'
        )

        combined = combined[combined['purchased'].isna()]
        combined = combined.drop(columns='purchased')
        result = (
            combined.groupby('customer_id', sort=False)
            .head(self.k)
            .reset_index(drop=True)
        )
        print("Generated style alternatives candidates")
        return result
