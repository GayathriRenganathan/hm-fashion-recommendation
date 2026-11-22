import pandas as pd
import numpy as np

class FeatureGenerator:
    def __init__(self, train_transactions_df, articles_df=None, customers_df=None):
        self.tx = train_transactions_df.copy()
        self.articles_df = articles_df
        self.customers_df = customers_df
        if self.articles_df is not None:
            cols_to_merge = [c for c in ['article_id', 'department_no', 'product_type_no'] if c in self.articles_df.columns]
            if cols_to_merge:
                self.tx = self.tx.merge(
                    self.articles_df[cols_to_merge].drop_duplicates(), 
                    on='article_id', 
                    how='left'
                )

    def add_item_features(self, df):
        """Merges article metadata and multi-window rolling popularities."""
        print("-> Adding item metadata and multi-window popularities...")
        
        if self.articles_df is not None:
            article_cols = [
                'article_id', 'product_type_no', 'graphical_appearance_no', 
                'colour_group_code', 'perceived_colour_value_id', 
                'department_no', 'index_group_no'
            ]
            available = [c for c in article_cols if c in self.articles_df.columns]
            df = df.merge(self.articles_df[available], on='article_id', how='left')


        # Multi-window historical popularity
        pop_windows = [1, 2, 4, 8]

        if not self.tx.empty:
            start_week = self.tx['week'].min()

            weekly_pop = (
                self.tx.groupby(['article_id', 'week'])
                .size()
                .unstack(fill_value=0)
                .reindex(
                    columns=range(start_week, start_week + max(pop_windows)),
                    fill_value=0
                )
            )

            pop_features = pd.DataFrame(index=weekly_pop.index)
            for window in pop_windows:
                pop_features[f'pop_{window}w'] = (
                    weekly_pop.iloc[:, :window].sum(axis=1)
                )

            pop_features = pop_features.reset_index()
            df = df.merge(pop_features, on='article_id', how='left')

        # Fill missing popularity values
        pop_cols = [f'pop_{window}w' for window in pop_windows]

        for col in pop_cols:
            if col not in df.columns:
                df[col] = 0
            else:
                df[col] = df[col].fillna(0)
                
        return df

    def add_user_features(self, df):
        """Merges customer demographics and historical purchase frequency."""
        print("-> Adding user demographics and profile features...")
        
        if self.customers_df is not None:
            customer_cols = ['customer_id', 'age', 'club_member_status', 'fashion_news_frequency']
            available = [c for c in customer_cols if c in self.customers_df.columns]
            df = df.merge(self.customers_df[available], on='customer_id', how='left')

        user_activity = self.tx.groupby('customer_id').agg(
            user_total_historical_purchases=('article_id', 'count'),
            user_last_purchase_week=('week', 'min')
        ).reset_index()
        df = df.merge(user_activity, on='customer_id', how='left')
        df['user_total_historical_purchases'] = df['user_total_historical_purchases'].fillna(0)
        df['user_last_purchase_week'] = df['user_last_purchase_week'].fillna(999)
        return df

    def add_user_item_affinity_features(self, df):
        """Calculates interaction history between a specific user and article/department."""
        print("-> Adding user-item interaction affinity features...")
        
        # Exact repeat purchase check: Has this user bought this exact article before?
        user_article_history = self.tx[['customer_id', 'article_id']].drop_duplicates()
        user_article_history['user_bought_article_before'] = 1
        
        df = df.merge(user_article_history, on=['customer_id', 'article_id'], how='left')
        df['user_bought_article_before'] = df['user_bought_article_before'].fillna(0).astype(int)

        # Category affinity: How many times has this user bought items from this department/product_type?
        if 'department_no' in df.columns and 'department_no' in self.tx.columns:
            # Merge department into transactions temporarily if not already there
            user_dept_pref = self.tx.groupby(['customer_id', 'department_no']).size().reset_index(name='user_dept_purchase_count')
            df = df.merge(user_dept_pref, on=['customer_id', 'department_no'], how='left')
            df['user_dept_purchase_count'] = df['user_dept_purchase_count'].fillna(0)
        # Product Type affinity 
        if 'product_type_no' in df.columns and 'product_type_no' in self.tx.columns:
            user_pt_pref = self.tx.groupby(['customer_id', 'product_type_no']).size().reset_index(name='user_product_type_purchase_count')
            df = df.merge(user_pt_pref, on=['customer_id', 'product_type_no'], how='left')
            df['user_product_type_purchase_count'] = df['user_product_type_purchase_count'].fillna(0)

        return df  


    def add_retrieval_metadata(self, df):
        """Preserves strategy source indicators."""
        print("-> Processing candidate retrieval flags and scores...")
        
        if 'source' in df.columns:
            sources = ['als', 'previous_purchases', 'item_similarity', 'recency', 'same_product_color', 'recency_popularity']
            for src in sources:
                df[f'is_from_{src}'] = (df['source'] == src).astype(int)
            
        return df

    def transform(self, candidates_df):
        """Executes the full feature pipeline."""
        df = candidates_df.copy()
        df = self.add_item_features(df)
        df = self.add_user_features(df)
        df = self.add_user_item_affinity_features(df)
        df = self.add_retrieval_metadata(df)
        
        print(f"Feature matrix successfully built. Final shape: {df.shape}")
        return df