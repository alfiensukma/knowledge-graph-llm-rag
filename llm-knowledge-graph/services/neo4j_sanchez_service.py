from __future__ import annotations
from typing import List, Dict, Optional, FrozenSet
from dataclasses import dataclass
import re
from langchain_neo4j import Neo4jGraph
from services.ontology_index_service import ICIndex, NodeId


@dataclass
class FrequentItemsetRow:
    items: FrozenSet[NodeId]
    support: int

class Neo4jIOService:
    def __init__(self, graph: Neo4jGraph, ic_index: ICIndex):
        self.graph = graph
        self.idx = ic_index


    def fetch_frequent_itemsets(self, min_support: int = 2):
        rows = self.graph.query("""
            MATCH (f:FrequentTopicSet)
            WHERE f.supportCount >= $min_support
            RETURN f.items AS items, f.supportCount AS sup
        """, {"min_support": min_support})

        out = []
        known_labels = set(self.idx.id2label.keys())

        for r in rows or []:
            raw_labels = r["items"] or []
            cleaned = [(lab or "").strip() for lab in raw_labels if lab]
            present = [lab for lab in cleaned if lab in known_labels]
            if len(present) >= 2:
                out.append(FrequentItemsetRow(items=frozenset(present), support=int(r["sup"])))
        return out