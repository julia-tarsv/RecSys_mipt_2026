"""EASE closed-form solution from scratch (dense item-item inverse)."""
import numpy as np
from scipy import sparse


class EASE:
    def __init__(self, l2=400.):
        self.l2 = float(l2)

    def fit(self, interactions):
        self.X = sparse.csr_matrix(interactions, dtype=np.float32).copy()
        self.X.data[:] = 1.
        G = (self.X.T @ self.X).toarray().astype(np.float64)
        G.flat[::G.shape[0]+1] += self.l2
        P = np.linalg.solve(G, np.eye(G.shape[0]))
        diag = np.diag(P).copy()
        self.B = -P / diag[None, :]
        np.fill_diagonal(self.B, 0.)
        return self

    def score_users(self, user_ids):
        return np.asarray(self.X[np.asarray(user_ids)] @ self.B)
