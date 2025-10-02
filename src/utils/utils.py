# src/utils/utils.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def top_k_similar(query_vec, matrix, k=10, exclude_self_index=None):
    sims = cosine_similarity(query_vec.reshape(1,-1), matrix)[0]
    if exclude_self_index is not None:
        sims[exclude_self_index] = -1
    idx = np.argsort(sims)[-k:][::-1]
    return idx, sims[idx]
