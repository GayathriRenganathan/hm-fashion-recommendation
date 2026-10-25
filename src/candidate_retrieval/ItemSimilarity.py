from src.candidate_retrieval.BaseCandidateGenerator import BaseCandidateGenerator
from typing import List
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.nearest_neighbours import CosineRecommender

class ItemSimilarityCandidates(BaseCandidateGenerator):

  def __init__(
      self,
      transactions: pd.DataFrame,
      articles: pd.DataFrame,
      lookback_weeks: int = 12,
      min_item_frequency: int = 5,  # Drop items bought by fewer than N users
      neighbors: int = 10,
      k: int = 30
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

    counts = (
        history.groupby(["item_idx", "user_idx"]).size().reset_index(name="freq")
    )
    user_item_matrix = csr_matrix(
        (np.ones(len(history), dtype=np.float32), (history["user_idx"], history["item_idx"])),
        shape=(n_users, n_items),
    )

    # 4. Compute item-item co-occurrence matrix (Items x Items)
    # Binary interaction dot product counts shared buyer overlaps instantly via C backend
    item_cooc = user_item_matrix.T.dot(user_item_matrix)
    item_cooc.setdiag(0)
    item_cooc.eliminate_zeros()

    # 5. Map target users to internal indices
    user_map = {uid: idx for idx, uid in enumerate(user_uniques)}
    valid_users = [(u, user_map[u]) for u in users if u in user_map]

    if not valid_users:
      return pd.DataFrame(columns=["customer_id", "article_id"])
    
    target_cust_ids = [t[0] for t in valid_users]
    target_indices = [t[1] for t in valid_users]

    # 3. MEMORY FIX: Slice the user-item matrix to ONLY the target users 
    # before performing the dot product, avoiding billions of temporary allocations.
    sub_user_item_matrix = user_item_matrix[target_indices, :]

    # 4. Vectorized scoring restricted strictly to target users
    user_scores = sub_user_item_matrix.dot(item_cooc)
    
    print("Generated candidate scores for users")

    # 7. Unpack results into a DataFrame
    recs_list = []
    for u_idx, cust_id in enumerate(target_cust_ids):
      row_start = user_scores.indptr[u_idx]
      row_end = user_scores.indptr[u_idx + 1]
      
      if row_start == row_end:
        continue

      col_indices = user_scores.indices[row_start:row_end]
      scores = user_scores.data[row_start:row_end]

      # Mask items the user already bought directly in sparse representation
      bought_start = user_item_matrix.indptr[u_idx]
      bought_end = user_item_matrix.indptr[u_idx + 1]
      bought_items = set(user_item_matrix.indices[bought_start:bought_end])

      # Filter out already bought items and sort top-k
      mask = ~np.isin(col_indices, list(bought_items))
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
        recs_list.append({
            "customer_id": cust_id,
            "article_id": item_uniques[item_idx],
        })

    return pd.DataFrame(recs_list)