from __future__ import annotations
from typing import Optional, Iterable, FrozenSet, List
from dataclasses import dataclass
import math
from services.ontology_index_service import ICIndex, NodeId

@dataclass
class LCSResult:
    lcs: Optional[NodeId]
    depth: int

class SanchezSimilarityService:
    def __init__(self, ic_index: ICIndex):
        self.idx = ic_index


    def lcs_deepest(self, c1: NodeId, c2: NodeId) -> LCSResult:
        A = self.idx.ancestors.get(c1, set())
        B = self.idx.ancestors.get(c2, set())
        inter = A & B
        if not inter:
            return LCSResult(None, -1)
        # select higher depth
        best = max(inter, key=lambda x: self.idx.depth.get(x, -10**9))
        return LCSResult(best, self.idx.depth.get(best, -1))


    def sim_lin(self, c1: NodeId, c2: NodeId, min_lcs_depth: Optional[int] = None) -> float:
        if c1 == c2:
            return 1.0
        lcs = self.lcs_deepest(c1, c2)
        if lcs.lcs is None:
            return 0.0
        if min_lcs_depth is not None and lcs.depth < min_lcs_depth:
            return 0.0
        ic1 = self.idx.ic.get(c1, 0.0)
        ic2 = self.idx.ic.get(c2, 0.0)
        ic_lcs = self.idx.ic.get(lcs.lcs, 0.0)
        denom = ic1 + ic2
        if denom <= 1e-12:
            return 0.0
        return max(0.0, min(1.0, (2.0 * ic_lcs) / denom))


    def sim_resnik(self, c1: NodeId, c2: NodeId) -> float:
        lcs = self.lcs_deepest(c1, c2)
        return 0.0 if lcs.lcs is None else self.idx.ic.get(lcs.lcs, 0.0)


    def dist_jiang_conrath(self, c1: NodeId, c2: NodeId) -> float:
        lcs = self.lcs_deepest(c1, c2)
        if lcs.lcs is None:
            return math.inf
        ic1 = self.idx.ic.get(c1, 0.0)
        ic2 = self.idx.ic.get(c2, 0.0)
        ic_lcs = self.idx.ic.get(lcs.lcs, 0.0)
        return max(0.0, ic1 + ic2 - 2.0 * ic_lcs)