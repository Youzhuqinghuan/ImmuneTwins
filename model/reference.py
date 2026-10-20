"""Personal composition and lineage-state references used after history adaptation."""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.special import softmax

DEFAULT_LAMBDAS = (.25, .5, 1., 2., 4., 8., 16., 32., 64., float("inf"))
LINEAGES = ("B", "CD4_T", "CD8_T", "NK", "Myeloid")

def clr(counts):
    """Count-space CLR with the paper's half-cell pseudocount."""
    values = np.log(np.asarray(counts, float) + .5)
    return values - values.mean(axis=-1, keepdims=True)

def scale_response(source, predicted, gamma):
    """Scale displacement in raw RNA coordinates without changing its direction."""
    if source.shape != predicted.shape:
        raise ValueError("Source and neural prediction must have identical row alignment")
    return source.copy() if gamma == 0 else predicted.copy() if gamma == 1 else np.asarray(source + gamma * (predicted - source), np.float32)

def resample_composition(labels, probability, classes, unit_id, seed=42):
    """Systematic cell resampling; retain labels and source-row provenance."""
    lookup = {label: i for i, label in enumerate(classes)}
    try:
        codes = np.array([lookup[label] for label in labels])
    except KeyError as error:
        raise ValueError(f"Unknown source cell type: {error}") from error
    counts = np.bincount(codes, minlength=len(classes))
    probability = np.asarray(probability, float)
    if probability.shape != (len(classes),) or not np.isfinite(probability).all() or (probability < 0).any():
        raise ValueError("Composition probabilities must be finite and nonnegative")
    missing_mass = float(probability[counts == 0].sum())
    weights = probability[codes] / counts[codes]
    if not weights.sum() > 0:
        raise ValueError("Forecast has no probability mass on observed source cell types")
    weights /= weights.sum()
    payload = "|".join(map(str, (seed, "resample", unit_id))).encode()
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(payload).digest()[:4], "little"))
    indices = np.searchsorted(np.cumsum(weights), (rng.random() + np.arange(len(labels))) / len(labels))
    return np.minimum(indices, len(labels) - 1), missing_mass

def apply_reference(source, predicted, labels, classes, forecast_clr, gamma, unit_id,
                    *, displacement=None, state_coefficient=0., coordinate_scale=None, seed=42):
    """Return final cells and their row indices."""
    cells = scale_response(source, predicted, gamma)
    if state_coefficient:
        if displacement is None or coordinate_scale is None:
            raise ValueError("Lineage-state correction requires a fitted reference and fold scale")
        cells = (cells + state_coefficient * displacement * coordinate_scale).astype(np.float32)
    indices, missing = resample_composition(labels, softmax(forecast_clr), classes, unit_id, seed)
    return cells[indices], indices, missing


class CompositionReference:
    """Donor-weighted CLR return model with nested shrinkage selection."""

    def __init__(self, counts, visits, inner_splits=None, lambda_grid=None):
        self.inner_splits = inner_splits
        self.lambda_grid = list(map(float, lambda_grid if lambda_grid is not None else DEFAULT_LAMBDAS))
        self.counts = counts
        self.V = visits
        self.X = pd.DataFrame(clr(counts), index=counts.index, columns=counts.columns)
        self.refs = {}
        self.history = {}
        for (src, v) in visits.iterrows():
            ix = visits.index[visits.donor.eq(v.donor) & visits.rest & visits.day.lt(v.day)].tolist()
            self.history[src] = ix
            self.refs[src] = (self.X.loc[ix].mean().values if ix else np.zeros(counts.shape[1]), len(ix))
        self.donor_mu = self.X[visits.rest].groupby(visits.loc[visits.rest, 'donor']).mean()

    def prior(self, donors, weights=None):
        ix = sorted(set(donors) & set(self.donor_mu.index))
        a = self.donor_mu.loc[ix].values
        return a.mean(0) if weights is None else np.average(a, axis=0, weights=[weights.get(d, 0) for d in ix])

    def arrays(self, U):
        return (self.X.loc[U.source].values, self.X.loc[U.target].values, np.stack([self.refs[s][0] for s in U.source]), np.array([self.refs[s][1] for s in U.source]))

    def fit(self, U, lam, mu=None):
        mu = self.prior(U.donor) if mu is None else mu
        (X, Y, R, N) = self.arrays(U)
        a = np.zeros(len(N)) if np.isinf(lam) else N / (N + lam)
        D = mu + a[:, None] * (R - mu) - X
        delta = Y - X
        w = 1 / U.donor.map(U.donor.value_counts()).values
        w = w / w.sum()
        ym = w @ delta
        yc = delta - ym
        K = float(np.sum(w[:, None] * D * yc) / max(np.sum(w[:, None] * D * D), 1e-12))
        m = ym
        return {'lambda': lam, 'K': K, 'm': m, 'mu': mu}

    def predict(self, U, p, ref_override=None):
        X = self.X.loc[U.source].values
        R = np.stack([self.refs[s][0] for s in U.source])
        N = np.array([self.refs[s][1] for s in U.source])
        if ref_override is not None:
            (R, N) = ref_override
        lam = p['lambda']
        a = np.zeros(len(N)) if np.isinf(lam) else N / (N + lam)
        return X + p['m'] + p['K'] * (p['mu'] + a[:, None] * (R - p['mu']) - X)

    def select(self, U, fold):
        split = self.inner_splits
        if split is None:
            raise ValueError('Composition fitting requires inner donor splits')
        dm = split[split.outer_fold.eq(fold)].set_index('donor_id').inner_fold
        if U.empty or U.donor.map(dm).isna().any():
            raise ValueError("Every fitting donor must have an inner fold")
        scores = []
        oofs = []
        for lam in self.lambda_grid:
            pred = np.full((len(U), self.X.shape[1]), np.nan)
            for j in range(4):
                mask = U.donor.map(dm).eq(j).values
                if not mask.any():
                    continue
                p = self.fit(U.loc[~mask], lam)
                pred[mask] = self.predict(U.loc[mask], p)
            err = np.sqrt(np.mean((pred - self.X.loc[U.target].values) ** 2, axis=1))
            scores.append(float(pd.Series(err, index=U.donor).groupby(level=0).mean().mean()))
            oofs.append(pred)
        j = min(range(len(scores)), key=lambda j: (round(scores[j], 12), -j))
        p = self.fit(U, self.lambda_grid[j])
        resid = (oofs[j] - self.X.loc[U.target].values) ** 2
        n = np.array([self.refs[s][1] for s in U.source])
        z = np.zeros(len(n)) if np.isinf(p['lambda']) else 1 / (n + p['lambda'])
        w = 1 / U.donor.map(U.donor.value_counts()).values
        design = np.column_stack([np.ones(len(n)), z])
        ab = nnls(design * np.sqrt(w[:, None]), resid.mean(1) * np.sqrt(w))[0]
        (p['a'], p['b']) = ab
        return (p, scores, oofs[j])

    def sd(self, n, p):
        n = np.asarray(n)
        term = 0 if np.isinf(p['lambda']) else p['b'] / (n + p['lambda'])
        return np.sqrt(p['a'] + term)

class StateReference:
    """Training-only batch offsets and standardized past lineage-state references."""

    def __init__(self, V, donors, lam, raw_means, center, scale):
        self.V = V
        self.donors = set(donors)
        self.lam = lam
        self.offsets = {}
        self.mu = {}
        self.audit = []
        self.M = {}
        mm = raw_means
        for L in LINEAGES:
            m = mm[mm.lineage.eq(L)].set_index('sample_id')[[f'mean_{j}' for j in range(16)]]
            m = m.loc[np.isfinite(m.values).all(axis=1)].copy()
            m = (m - center) / scale
            self.M[L] = m
            meta = V.loc[m.index]
            train = meta.donor.isin(self.donors)
            rest = train & meta.rest
            donor_center = m.loc[rest].groupby(meta.loc[rest, 'donor']).mean()
            population = donor_center.mean().values
            if not np.isfinite(population).all():
                raise ValueError(('invalid state prior', L, 'reference'))
            samples = m.loc[train].copy()
            mtrain = meta.loc[train]
            cent = donor_center.reindex(mtrain.donor).fillna(pd.Series(population, index=m.columns)).values
            residual = samples.values - cent
            frame = pd.DataFrame(residual, columns=range(16))
            frame['donor'] = mtrain.donor.values
            frame['batch'] = mtrain.batch.values
            db = frame.groupby(['donor', 'batch'])[list(range(16))].mean().reset_index()
            grouped = db.groupby('batch')[list(range(16))]
            batchmean = grouped.mean()
            n = grouped.size()
            within = grouped.var().fillna(pd.Series(np.var(residual, axis=0, ddof=1), index=range(16)))
            sampling = within.div(n, axis=0)
            grand = np.average(batchmean.values, axis=0, weights=n.values)
            tau = np.maximum(np.average((batchmean.values - grand) ** 2 - sampling.values, axis=0, weights=n.values), 0)
            w = np.divide(tau, tau + sampling.values, out=np.zeros_like(sampling.values), where=tau + sampling.values > 1e-12)
            offset = (batchmean.values - grand) * w
            self.offsets[L] = {b: off for (b, off) in zip(batchmean.index, offset)}
            corrected = np.stack([x - self.offsets[L].get(meta.loc[s, 'batch'], np.zeros(16)) for (s, x) in m.loc[rest].iterrows()])
            self.mu[L] = pd.DataFrame(corrected, index=meta.loc[rest, 'donor']).groupby(level=0).mean().mean().values
            for (b, ns, ww, oo) in zip(batchmean.index, n, w, offset):
                self.audit.append(dict(lineage=L, batch=b, n_donors=int(ns), shrinkage=float(ww.mean()), offset_norm=float(np.linalg.norm(oo)), tau2_mean=float(tau.mean())))

    def displacement(self, source, donor, lineages):
        assert donor not in self.donors, 'query donor entered state reference fit'
        v = self.V.loc[source]
        h = self.V.index[self.V.donor.eq(donor) & self.V.rest & self.V.day.lt(v.day)]
        disps = {}
        ns = {}
        for (L, m) in self.M.items():
            ix = [s for s in h if s in m.index and np.isfinite(m.loc[s]).all()]
            off = self.offsets[L]
            if source not in m.index or not np.isfinite(m.loc[source]).all():
                disps[L] = np.zeros(16)
                ns[L] = 0
                continue
            src = m.loc[source].values - off.get(v.batch, np.zeros(16))
            n = len(ix)
            a = 0 if np.isinf(self.lam) else n / (n + self.lam)
            ref = np.mean([m.loc[s].values - off.get(self.V.loc[s, 'batch'], np.zeros(16)) for s in ix], axis=0) if n else self.mu[L]
            disps[L] = self.mu[L] + a * (ref - self.mu[L]) - src
            ns[L] = n
        return (np.stack([disps[L] for L in lineages]), ns)
