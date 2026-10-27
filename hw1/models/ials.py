import numpy as np
from scipy import sparse


class ImplicitALS:
    def __init__(self, n_factors=24, reg=0.1, alpha=20., n_iters=8, seed=42):
        self.n_factors = int(n_factors)
        self.reg = float(reg)
        self.alpha = float(alpha)
        self.n_iters = int(n_iters)
        self.seed = int(seed)

    def fit(self, counts, targets=None):
        counts = sparse.csr_matrix(counts, dtype=np.float64)
        counts.sort_indices()
        self.counts = counts
        if targets is None:
            goal = counts.copy()
            goal.data[:] = 1.
        else:
            goal = sparse.csr_matrix(targets, dtype=np.float64)
            goal.sort_indices()
            if not np.array_equal(counts.indptr, goal.indptr) or not np.array_equal(counts.indices, goal.indices):
                raise ValueError('targets must have exactly the same sparse support as counts')
            if np.any(goal.data < 0) or np.any(goal.data > 1):
                raise ValueError('targets must be within [0, 1]')
        confidence = counts.copy()
        confidence.data = 1.0 + self.alpha * np.log1p(confidence.data)
        rng = np.random.default_rng(self.seed)
        m, n = counts.shape
        self.user_factors = rng.normal(0, .02, (m, self.n_factors))
        self.item_factors = rng.normal(0, .02, (n, self.n_factors))
        for _ in range(self.n_iters):
            self.user_factors = self._update(confidence, goal, self.item_factors)
            self.item_factors = self._update(confidence.T.tocsr(), goal.T.tocsr(), self.user_factors)
        return self

    def _update(self, conf, goal, fixed):
        gram = fixed.T @ fixed
        eye = np.eye(self.n_factors)
        out = np.empty((conf.shape[0], self.n_factors), dtype=np.float64)
        for row in range(conf.shape[0]):
            lo, hi = conf.indptr[row:row+2]
            ids = conf.indices[lo:hi]
            c = conf.data[lo:hi]
            p = goal.data[lo:hi]
            V = fixed[ids]
            A = gram + (V.T * (c - 1.)) @ V + self.reg * eye
            b = V.T @ (c * p)
            out[row] = np.linalg.solve(A, b)
        return out

    def score_users(self, user_ids):
        return self.user_factors[np.asarray(user_ids)] @ self.item_factors.T
