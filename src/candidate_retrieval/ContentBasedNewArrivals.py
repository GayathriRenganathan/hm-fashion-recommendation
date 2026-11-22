import pandas as pd
from typing import List

class ContentBasedNewArrivalsCandidates:
    def __init__(
        self, 
        transactions: pd.DataFrame, 
        articles: pd.DataFrame, 
        lookback_weeks: int = 4, 
        k: int = 20
    ):
        """
        Generates content-based new arrival candidates using a two-tiered routing strategy:
        - Low-history users (< 3 total purchases): Preferred Department New Arrivals
        - Deep-history users (>= 3 total purchases): Preferred Product Type New Arrivals
        """
        self.lookback_weeks = lookback_weeks
        self.k = k
        
        print("Precomputing tiered content-based new arrival candidates...")
        recent_tx = transactions[transactions['week'] <= self.lookback_weeks]
        recent_articles = recent_tx[['article_id']].drop_duplicates().merge(
            articles[['article_id', 'department_no', 'product_type_no']], 
            on='article_id', 
            how='inner'
        )
        # Merge article metadata to transactions
        tx_full = transactions.merge(
            articles[['article_id', 'department_no', 'product_type_no']], 
            on='article_id', 
            how='inner'
        )
        
        # 1. Classify users by total purchase history length
        user_total_counts = tx_full.groupby('customer_id', observed=True).size().reset_index(name='total_purchases')
        self.user_purchase_counts_dict = dict(zip(user_total_counts['customer_id'], user_total_counts['total_purchases']))
        self.low_history_users = set(user_total_counts[user_total_counts['total_purchases'] < 3]['customer_id'])
        
        # Split transactions by user tier
        is_low = tx_full['customer_id'].isin(self.low_history_users)
        low_history_tx = tx_full[is_low]
        deep_history_tx = tx_full[~is_low]

        # 2. Low-history routing: Find favorite department safely
        low_dept_counts = low_history_tx.groupby(['customer_id', 'department_no'], observed=True).size().reset_index(name='count')
        fav_dept_df = (
            low_dept_counts.sort_values(['customer_id', 'count'], ascending=[True, False])
            .drop_duplicates(subset=['customer_id'])
        )
        self.user_fav_dept = dict(zip(fav_dept_df['customer_id'], fav_dept_df['department_no']))
        
        # 3. Deep-history routing: Find favorite product type safely
        deep_pt_counts = deep_history_tx.groupby(['customer_id', 'product_type_no'], observed=True).size().reset_index(name='count')
        fav_pt_df = (
            deep_pt_counts.sort_values(['customer_id', 'count'], ascending=[True, False])
            .drop_duplicates(subset=['customer_id'])
        )
        self.user_fav_product_type = dict(zip(fav_pt_df['customer_id'], fav_pt_df['product_type_no']))

        # 4. Precompute top-K newest items per department and product type as dictionaries (NO MERGES!)
        articles_sorted_dept = recent_articles.sort_values(['department_no', 'article_id'], ascending=[True, False])
        self.dept_top_k_dict = articles_sorted_dept.groupby('department_no', observed=True)['article_id'].apply(lambda x: x.head(self.k).tolist()).to_dict()
        
        articles_sorted_pt = recent_articles.sort_values(['product_type_no', 'article_id'], ascending=[True, False])
        self.pt_top_k_dict = articles_sorted_pt.groupby('product_type_no', observed=True)['article_id'].apply(lambda x: x.head(self.k).tolist()).to_dict()
        
        # 5. Global fallback: Absolute newest items overall
        self.global_newest = recent_articles.sort_values('article_id', ascending=False)['article_id'].tolist()[:self.k]
        
        print("Tiered content-based precomputation complete!")

    def generate(self, users: List[str]) -> pd.DataFrame:
        """
        Generates top-k tiered content-based candidates for a batch of users.
        """
        if len(users) == 0:
            return pd.DataFrame(columns=["customer_id", "article_id"])
            
        rows = []
        for uid in users:
            total_purchases = self.user_purchase_counts_dict.get(uid, 0)
            
            if total_purchases < 3:
                # Tier 1: Department routing
                dept = self.user_fav_dept.get(uid)
                items = self.dept_top_k_dict.get(dept, self.global_newest) if dept is not None else self.global_newest
            else:
                # Tier 2: Product Type routing
                pt = self.user_fav_product_type.get(uid)
                items = self.pt_top_k_dict.get(pt, self.global_newest) if pt is not None else self.global_newest
                
            for item_id in items[:self.k]:
                rows.append({"customer_id": uid, "article_id": item_id})
                
        return pd.DataFrame(rows, columns=["customer_id", "article_id"])