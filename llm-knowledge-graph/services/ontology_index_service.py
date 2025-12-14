from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict, deque
import math

from langchain_neo4j import Neo4jGraph

NodeId = str

@dataclass
class ICIndex:
    ic: Dict[NodeId, float]
    depth: Dict[NodeId, int]
    ancestors: Dict[NodeId, Set[NodeId]]
    descendants: Dict[NodeId, Set[NodeId]]
    leaves_below: Dict[NodeId, int]
    num_subsumers: Dict[NodeId, int]
    id2label: Dict[NodeId, str]
    label2id: Dict[str, NodeId]
    max_leaves: int
    max_subsumers: int
    roots: Set[NodeId]

class OntologyIndexService:
    def __init__(self, graph: Neo4jGraph, max_depth: Optional[int] = None):
        self.graph = graph
        self.max_depth = max_depth


    def _fetch_topics(self) -> Tuple[Dict[NodeId, str], Dict[str, NodeId]]:
        rows = self.graph.query("""
            MATCH (t:Topic)
            WHERE t.label <> 'computer science'
            RETURN t.label AS id, coalesce(t.label_norm, t.label) AS label_norm
        """)
        id2label = {r["id"]: r["label_norm"] for r in rows}
        label2id = {v: k for k, v in id2label.items()}
        return id2label, label2id


    def _fetch_edges(self) -> List[Tuple[NodeId, NodeId]]:
        rows = self.graph.query("""
            MATCH (sub:Topic)-[:SUB_TOPIC_OF]->(sup:Topic)
            WHERE sub.label <> 'computer science' AND sup.label <> 'computer science'
            RETURN sub.label AS sub, sup.label AS sup
        """)
        return [(r["sub"], r["sup"]) for r in rows]


    @staticmethod
    def _topo_order(nodes: Set[NodeId], parents, children):
        indeg = {n: len(parents[n]) for n in nodes}
        q = deque([n for n in nodes if indeg[n] == 0])
        topo = []
        while q:
            u = q.popleft()
            topo.append(u)
            for v in children[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        return topo, indeg


    @staticmethod
    def _build_adj(nodes: Set[NodeId], edges: List[Tuple[NodeId, NodeId]]):
        parents = defaultdict(set)   # child -> set(parents)
        children = defaultdict(set)  # parent -> set(children)
        for u, v in edges:
            if u in nodes and v in nodes:
                parents[u].add(v)   # u (child) has parent v
                children[v].add(u)  # v (parent) has child u
        for n in nodes:
            parents.setdefault(n, set())
            children.setdefault(n, set())
        return parents, children


    @staticmethod
    def _find_roots(nodes: Set[NodeId], parents: Dict[NodeId, Set[NodeId]]) -> Set[NodeId]:
        return {n for n in nodes if not parents[n]}


    @staticmethod
    def _topo_depth(roots: Set[NodeId], children: Dict[NodeId, Set[NodeId]]) -> Dict[NodeId, int]:
        depth = {r: 0 for r in roots}
        q = deque(roots)
        while q:
            u = q.popleft()
            for v in children[u]:
                nd = depth[u] + 1
                if v not in depth or nd < depth[v]:
                    depth[v] = nd
                    q.append(v)
        return depth


    @staticmethod
    def _compute_anc_desc_iter(nodes: Set[NodeId], parents, children, topo: List[NodeId]):
        ancestors: Dict[NodeId, Set[NodeId]] = {n: {n} for n in nodes}
        descendants: Dict[NodeId, Set[NodeId]] = {n: set() for n in nodes}
        # propagate ancestors
        for u in topo:
            for v in children[u]:
                ancestors[v] |= ancestors[u]
        # propagate descendants
        for u in reversed(topo):
            for p in parents[u]:
                descendants[p].add(u)
                descendants[p] |= descendants[u]

        return ancestors, descendants


    @staticmethod
    def _compute_leaves_below_iter(nodes: Set[NodeId], children, topo: List[NodeId]) -> Dict[NodeId, int]:
        leaves_below = {n: 0 for n in nodes}
        # node without child = leaf -> 1
        for n in nodes:
            if not children[n]:
                leaves_below[n] = 1
        # accumulation from bottom to top
        for u in reversed(topo):
            if children[u]:
                leaves_below[u] = sum(leaves_below[v] for v in children[u])
        return leaves_below


    @staticmethod
    def _ic_sanchez_norm(
        leaves_c: int,
        max_leaves: int,
        num_subsumers_c: int,
        max_subsumers: int,
        eps: float = 1e-12
    ) -> float:
        num = -math.log((leaves_c + 1) / (max_leaves + 1) + eps)
        den = math.log(max_leaves + 1 + eps)
        leaf_term = (num / den) if den > 0 else 0.0  # [0,1]
        subs_num = math.log(num_subsumers_c + 1 + eps)
        subs_den = math.log(max_subsumers + 1 + eps)
        subs_term = (subs_num / subs_den) if subs_den > 0 else 0.0  # [0,1]
        ic = max(0.0, min(1.0, leaf_term * subs_term))
        return ic


    def build_ic_index(self) -> ICIndex:
        id2label, label2id = self._fetch_topics()
        edges = self._fetch_edges()
        nodes: Set[NodeId] = set(id2label.keys())
        parents, children = self._build_adj(nodes, edges)
        roots = self._find_roots(nodes, parents)
        # Topo order
        topo, indeg_residual = self._topo_order(nodes, parents, children)
        if len(topo) != len(nodes):
            cyclic = [n for n, d in indeg_residual.items() if d > 0][:50]
            labels = [id2label.get(n, n) for n in cyclic]
            raise RuntimeError(
                "Ontology has cycles in SUB_TOPIC_OF; please fix in Neo4j. "
                f"Sample nodes in cycles: {labels}"
            )
        # Depth from root
        depth = {r: 0 for r in roots}
        for u in topo:
            for v in children[u]:
                nd = depth.get(u, 0) + 1
                depth[v] = nd if v not in depth else min(depth[v], nd)   
        # Ancestors/descendants
        ancestors, descendants = self._compute_anc_desc_iter(nodes, parents, children, topo)
        leaves_below = self._compute_leaves_below_iter(nodes, children, topo)
        # IC
        num_subsumers = {n: len(ancestors[n]) - 1 for n in nodes}
        max_leaves = max(leaves_below.values()) if leaves_below else 1
        max_subsumers = max(num_subsumers.values()) if num_subsumers else 1
        ic = {
            n: self._ic_sanchez_norm(
                leaves_below[n], max_leaves,
                num_subsumers[n], max_subsumers
            )
            for n in nodes
        }

        return ICIndex(
            ic=ic,
            depth=depth,
            ancestors=ancestors,
            descendants=descendants,
            leaves_below=leaves_below,
            num_subsumers=num_subsumers,
            id2label=id2label,
            label2id=label2id,
            max_leaves=max_leaves,
            max_subsumers=max_subsumers,
            roots=roots
        )
