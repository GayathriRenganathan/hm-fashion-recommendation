from src.candidate_retrieval.BaseCandidateGenerator import BaseCandidateGenerator
from typing import List
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


class ItemSimilarityCandidates(BaseCandidateGenerator):
    """
    Generates item similarity-based recommendation candidates using
    item-item collaborative filtering.

    Builds a customer-item interaction matrix from historical transactions
    and computes item-item co-occurrence scores based on shared customers.

    Customer recommendation scores are calculated by
    aggregating the co-occurrence scores of items in their purchase history.

    Previously purchased items are excluded, and the top-K candidates
    are returned for each customer.
    """

    def __init__(
        self,
        transactions: pd.DataFrame,
        articles: pd.DataFrame,
        lookback_weeks: int = 12,
        min_item_frequency: int = 5,  # Drop items bought by fewer than N users
        neighbors: int = 10,
        k: int = 30,
    ):
        super().__init__(transactions, articles)
        self.lookback_weeks = lookback_weeks
        self.min_item_frequency = min_item_frequency
        self.neighbors = neighbors
        self.k = k

    def generate(self, users: List[str]) -> pd.DataFrame:
        # Filter transactions to the lookback window
        history = self.transactions[
            self.transactions["week"] <= self.lookback_weeks
        ].copy()

        if history.empty:
            return pd.DataFrame(columns=["customer_id", "article_id"])
        history = history.drop_duplicates(["customer_id", "article_id"])

        # Remove rare items
        item_counts = history["article_id"].value_counts()
        valid_items = item_counts[item_counts >= self.min_item_frequency].index
        history = history[history["article_id"].isin(valid_items)]

        if history.empty:
            return pd.DataFrame(columns=["customer_id", "article_id"])

        history["user_idx"], user_uniques = pd.factorize(history["customer_id"])
        history["item_idx"], item_uniques = pd.factorize(history["article_id"])

        n_users = len(user_uniques)
        n_items = len(item_uniques)

        user_item_matrix = csr_matrix(
            (np.ones(len(history), dtype=np.float32), (history["user_idx"], history["item_idx"])),
            shape=(n_users, n_items),
        )

        # Compute item-item co-occurrence matrix (Items x Items)
        # Binary interaction dot product counts shared buyer overlaps instantly via C backend
        item_cooc = user_item_matrix.T.dot(user_item_matrix)
        item_cooc.setdiag(0)
        item_cooc.eliminate_zeros()

        # # Keep only top-N neighbors for each item
        # pruned_indices = []
        # pruned_data = []
        # pruned_indptr = [0]

        # for i in range(n_items):
        #     start = item_cooc.indptr[i]
        #     end = item_cooc.indptr[i + 1]

        #     row_indices = item_cooc.indices[start:end]
        #     row_data = item_cooc.data[start:end]

        #     if len(row_data) > self.neighbors:
        #         top_idx = np.argpartition(
        #             row_data, -self.neighbors
        #         )[-self.neighbors:]

        #         row_indices = row_indices[top_idx]
        #         row_data = row_data[top_idx]

        #     pruned_indices.extend(row_indices)
        #     pruned_data.extend(row_data)
        #     pruned_indptr.append(len(pruned_indices))

        # item_cooc = csr_matrix(
        #     (
        #         np.asarray(pruned_data, dtype=np.float32),
        #         np.asarray(pruned_indices, dtype=np.int32),
        #         np.asarray(pruned_indptr, dtype=np.int32)
        #     ),
        #     shape=(n_items, n_items)
        # )

        # print(f"Pruned item-item relationships to top {self.neighbors} neighbors")

        # Map target users to internal indices
        user_map = {uid: idx for idx, uid in enumerate(user_uniques)}
        valid_users = [(u, user_map[u]) for u in users if u in user_map]

        if not valid_users:
            return pd.DataFrame(columns=["customer_id", "article_id"])

        target_cust_ids = [t[0] for t in valid_users]
        target_indices = [t[1] for t in valid_users]

        sub_user_item_matrix = user_item_matrix[target_indices, :]
        user_scores = sub_user_item_matrix.dot(item_cooc)

        print("Generated candidate scores for users")

        recs_list = []
        for u_idx, cust_id in enumerate(target_cust_ids):
            row_start = user_scores.indptr[u_idx]
            row_end = user_scores.indptr[u_idx + 1]

            if row_start == row_end:
                continue

            col_indices = user_scores.indices[row_start:row_end]
            scores = user_scores.data[row_start:row_end]

            bought_items = user_item_matrix.indices[
                user_item_matrix.indptr[target_indices[u_idx]] : user_item_matrix.indptr[
                    target_indices[u_idx] + 1
                ]
            ]

            mask = ~np.isin(col_indices, bought_items)
            filtered_indices = col_indices[mask]
            filtered_scores = scores[mask]

            if len(filtered_indices) == 0:
                continue

            # Get top-k using partition for maximum speed
            top_n = min(self.k, len(filtered_indices))
            top_relative_idx = np.argpartition(filtered_scores, -top_n)[-top_n:]
            top_relative_idx = top_relative_idx[np.argsort(filtered_scores[top_relative_idx])[::-1]]

            for idx in top_relative_idx:
                item_idx = filtered_indices[idx]
                recs_list.append(
                    {
                        "customer_id": cust_id,
                        "article_id": item_uniques[item_idx],
                    }
                )

        print("Generated item similarity candidates")
        return pd.DataFrame(recs_list, columns=["customer_id", "article_id"])