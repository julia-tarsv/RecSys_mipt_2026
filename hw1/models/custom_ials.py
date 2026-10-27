import numpy as np
from scipy import sparse
from .ials import ImplicitALS


class CustomTargetIALS(ImplicitALS):
    def __init__(self, half_life_days=90., **kwargs):
        super().__init__(**kwargs)
        self.half_life_days = float(half_life_days)

    def fit(self, counts, quantities, last_purchase, reference_time):
        counts = sparse.csr_matrix(counts, dtype=np.float64)
        quantities = sparse.csr_matrix(quantities, dtype=np.float64)
        last_purchase = sparse.csr_matrix(last_purchase, dtype=np.float64)
        for x in (quantities, last_purchase):
            if not (np.array_equal(x.indptr, counts.indptr) and np.array_equal(x.indices, counts.indices)):
                raise ValueError('All sparse inputs must have the same stored coordinates')
        c, q, t = counts.data, quantities.data, last_purchase.data
        p95_c = max(float(np.percentile(c, 95)), 1.)
        p95_q = max(float(np.percentile(q, 95)), 1.)
        frequency = np.clip(np.log1p(c) / np.log1p(p95_c), 0, 1)
        volume = np.clip(np.log1p(q) / np.log1p(p95_q), 0, 1)
        strength = .65 * frequency + .35 * volume
        age_days = np.maximum(0., (float(reference_time) - t) / 86400.)
        freshness = .5 + .5 * np.exp2(-age_days / self.half_life_days)
        signal = (.15 + .85 * strength) * freshness
        y = counts.copy()
        y.data = np.clip(signal, 0, 1)
        self.training_signal_summary = (float(y.data.min()), float(y.data.mean()), float(y.data.max()))
        return super().fit(counts, targets=y)
