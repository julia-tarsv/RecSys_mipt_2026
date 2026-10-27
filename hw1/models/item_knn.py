import numpy as np
from scipy import sparse

class ItemKNN:
    def __init__(self, n_neighbors=60, shrink=20.0):
        self.n_neighbors = int(n_neighbors)
        self.shrink = float(shrink)

    def fit(self, interactions):
        X = interactions.tocsr().astype(np.float32).copy()
        X.data[:] = 1.0
        self.X = X
        co = (X.T @ X).toarray()
        counts = np.diag(co).copy()
        denominator = np.sqrt(counts[:, None] * counts[None, :]) + self.shrink
        sim = np.divide(co, denominator, out=np.zeros_like(co), where=denominator > 0)
        np.fill_diagonal(sim, 0.)
        n = sim.shape[0]
        k = min(self.n_neighbors, max(0, n - 1))
        if k < n:
            idx = np.argpartition(sim, -k, axis=0)[:-k, :] if k else np.arange(n)[:, None]
            if k:
                sim[idx, np.arange(n)] = 0.
        self.S = sparse.csr_matrix(sim)
        return self

    def score_users(self, user_ids):
        return (self.X[np.asarray(user_ids)] @ self.S).toarray()
