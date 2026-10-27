import re
from collections import Counter
import numpy as np
import torch
from torch import nn
from scipy import sparse

STOPWORDS = {'THE', 'AND', 'OF', 'FOR', 'WITH', 'IN', 'A', 'AN', 'SET', 'PACK',
             'PCS', 'PC', 'X', 'ON', 'TO', 'AS', 'BY', 'SMALL', 'LARGE'}


def title_tokens(text):
    words = [x for x in re.findall(r'[A-Z]{3,}', str(text).upper()) if x not in STOPWORDS]
    return (words + ['UNKNOWN', 'UNKNOWN'])[:2]


class MetadataEncoder:
    def __init__(self, history, user_lookup, item_lookup, max_tokens=500):
        self.n_users = len(user_lookup)
        self.n_items = len(item_lookup)
        country = history.groupby('CustomerID')['Country'].agg(lambda s: s.mode().iloc[0])
        desc = history.groupby('StockCode')['Description'].agg(lambda s: s.mode().iloc[0])
        prices = history.groupby('StockCode')['UnitPrice'].median()
        country_vocab = {x: j for j, x in enumerate(sorted(country.dropna().unique()))}
        self.user_country = np.zeros(self.n_users, dtype=np.int64)
        for u, ix in user_lookup.items():
            self.user_country[ix] = country_vocab.get(country.get(u, ''), len(country_vocab))
        self.item_price = np.zeros(self.n_items, dtype=np.int64)
        item_words = {}
        freq = Counter()
        for it, ix in item_lookup.items():
            self.item_price[ix] = np.digitize(float(prices.get(it, 0.)), [1., 3., 6., 12., 25.])
            item_words[ix] = title_tokens(desc.get(it, ''))
            freq.update(item_words[ix])
        keywords = [word for word, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:max_tokens]]
        vocab = {word: j + 1 for j, word in enumerate(keywords)}  # 0=UNKNOWN
        self.item_keywords = np.zeros((self.n_items, 2), dtype=np.int64)
        for ix, words in item_words.items():
            self.item_keywords[ix] = [vocab.get(word, 0) for word in words]
        self.offset_user = 0
        self.offset_item = self.n_users
        self.offset_country = self.offset_item + self.n_items
        self.offset_price = self.offset_country + len(country_vocab) + 1
        self.offset_token = self.offset_price + 6
        self.n_features = self.offset_token + len(vocab) + 1
        self.n_fields = 6
        self.n_keyword_values = len(vocab) + 1

    def encode(self, users, items):
        users = np.asarray(users, dtype=np.int64)
        items = np.asarray(items, dtype=np.int64)
        if users.shape != items.shape:
            raise ValueError('users/items should have identical shapes')
        return np.column_stack((
            users + self.offset_user,
            items + self.offset_item,
            self.user_country[users] + self.offset_country,
            self.item_price[items] + self.offset_price,
            self.item_keywords[items, 0] + self.offset_token,
            self.item_keywords[items, 1] + self.offset_token
        )).astype(np.int64)


class _TorchFM(nn.Module):
    def __init__(self, n_features, n_factors):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0., dtype=torch.float32))
        self.linear = nn.Embedding(n_features, 1)
        self.latent = nn.Embedding(n_features, n_factors)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.latent.weight, mean=0, std=0.02)

    def forward(self, x):
        V = self.latent(x)
        interaction = .5 * ((V.sum(dim=1) ** 2 - (V ** 2).sum(dim=1)).sum(dim=1))
        return self.bias + self.linear(x).sum(dim=(1, 2)) + interaction


class FactorizationMachine:
    def __init__(self, metadata, n_factors=16, learning_rate=.01, epochs=4,
                 negatives=2, batch_size=8192, seed=42, weight_decay=1e-6):
        self.meta = metadata
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.negatives = negatives
        self.batch_size = batch_size
        self.seed = seed
        self.weight_decay = weight_decay

    def fit(self, interactions):
        X = sparse.csr_matrix(interactions)
        X.sort_indices()
        coo = X.tocoo()
        pu, pi = coo.row.astype(np.int64), coo.col.astype(np.int64)
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        self.model = _TorchFM(self.meta.n_features, self.n_factors)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss()
        for _ in range(self.epochs):
            has_unknown_items = np.diff(X.indptr)[pu] < X.shape[1]
            nu = np.repeat(pu[has_unknown_items], self.negatives)
            ni = rng.integers(0, X.shape[1], len(nu), dtype=np.int64)
            bad = np.asarray(X[nu, ni]).reshape(-1) > 0
            while bad.any():
                ni[bad] = rng.integers(0, X.shape[1], bad.sum(), dtype=np.int64)
                bad = np.asarray(X[nu, ni]).reshape(-1) > 0
            users = np.concatenate([pu, nu])
            items = np.concatenate([pi, ni])
            targets = np.r_[np.ones(len(pu), dtype=np.float32),
                            np.zeros(len(nu), dtype=np.float32)]
            order = rng.permutation(len(users))
            self.model.train()
            for ids in np.array_split(order, max(1, int(np.ceil(len(order) / self.batch_size)))):
                features = torch.from_numpy(self.meta.encode(users[ids], items[ids]))
                label = torch.from_numpy(targets[ids])
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(features)
                loss = criterion(logits, label)
                loss.backward()
                optimizer.step()
        self.model.eval()
        return self

    @torch.no_grad()
    def score_users(self, user_ids):
        us = np.asarray(user_ids, dtype=np.int64)
        n_items = self.meta.n_items
        out = np.empty((len(us), n_items), dtype=np.float32)
        for r, user in enumerate(us):
            for i in range(0, n_items, 4096):
                it = np.arange(i, min(i + 4096, n_items))
                features = torch.from_numpy(self.meta.encode(np.full(len(it), user), it))
                out[r, i:i + len(it)] = self.model(features).numpy()
        return out

class IDOnlyEncoder:
    def __init__(self, n_users, n_items):
        self.n_users, self.n_items = int(n_users), int(n_items)
        self.n_features = self.n_users + self.n_items
        self.n_fields = 2

    def encode(self, users, items):
        users = np.asarray(users, dtype=np.int64)
        items = np.asarray(items, dtype=np.int64)
        return np.column_stack([users, items + self.n_users])
