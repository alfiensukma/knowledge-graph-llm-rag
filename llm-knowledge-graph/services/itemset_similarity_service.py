from __future__ import annotations
from typing import FrozenSet, Optional, List
from dataclasses import dataclass
import numpy as np
from services.ontology_index_service import ICIndex, NodeId
from services.sanchez_similarity_service import SanchezSimilarityService

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

@dataclass
class SetSimConfig:
    method: str = "bma"  # "bma" | "hungarian"
    weight_by_ic: bool = False
    min_lcs_depth: Optional[int] = None


class ItemsetSimilarityService:
    def __init__(self, ic_index: ICIndex):
        self.idx = ic_index
        self.sanchez = SanchezSimilarityService(ic_index)


    def _max_sim_row(self, A: List[NodeId], B: List[NodeId], cfg: SetSimConfig) -> np.ndarray:
        sims = np.zeros((len(A), len(B)), dtype=float)
        for i, a in enumerate(A):
            for j, b in enumerate(B):
                sims[i, j] = self.sanchez.sim_lin(a, b, cfg.min_lcs_depth)
        return sims


    def _weights(self, S: List[NodeId], weight_by_ic: bool) -> np.ndarray:
        if not weight_by_ic:
            return np.ones(len(S), dtype=float)
        w = np.array([self.idx.ic.get(x, 0.0) for x in S], dtype=float)
        s = w.sum()
        return (w / s) if s > 0 else np.ones(len(S), dtype=float)


    def bma_similarity(self, A: FrozenSet[NodeId], B: FrozenSet[NodeId], cfg: SetSimConfig) -> float:
        if not A or not B:
            return 0.0
        A_l, B_l = list(A), list(B)
        M = self._max_sim_row(A_l, B_l, cfg)
        wA = self._weights(A_l, cfg.weight_by_ic)
        wB = self._weights(B_l, cfg.weight_by_ic)
        # A->B
        max_AB = M.max(axis=1)
        score_AB = float((wA * max_AB).sum())
        # B->A
        max_BA = M.max(axis=0)
        score_BA = float((wB * max_BA).sum())
        # normalization
        return 0.5 * (score_AB + score_BA)


    def hungarian_similarity(self, A: FrozenSet[NodeId], B: FrozenSet[NodeId], cfg: SetSimConfig) -> float:
        if not A or not B:
            return 0.0
        if not SCIPY_OK:
            # fallback
            return self.bma_similarity(A, B, cfg)

        A_l, B_l = list(A), list(B)
        M = self._max_sim_row(A_l, B_l, cfg)
        # Hungarian = cost minimization
        cost = 1.0 - M
        # pad matrix to square
        n, m = cost.shape
        if n > m:
            pad = np.ones((n, n - m))
            cost = np.hstack([cost, pad])
        elif m > n:
            pad = np.ones((m - n, m))
            cost = np.vstack([cost, pad])

        row_ind, col_ind = linear_sum_assignment(cost)
        pairs = []
        for i, j in zip(row_ind, col_ind):
            if i < len(A_l) and j < len(B_l):
                pairs.append((i, j))
        if not pairs:
            return 0.0
        sim_vals = [M[i, j] for (i, j) in pairs]
        # match average
        return float(np.mean(sim_vals))


    def set_similarity(self, A: FrozenSet[NodeId], B: FrozenSet[NodeId], cfg: SetSimConfig) -> float:
        if cfg.method == "hungarian":
            return self.hungarian_similarity(A, B, cfg)
        return self.bma_similarity(A, B, cfg)