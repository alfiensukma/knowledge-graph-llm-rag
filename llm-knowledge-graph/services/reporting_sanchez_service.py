from __future__ import annotations
from typing import List, Dict
import pandas as pd
from services.ontology_index_service import ICIndex
from services.itemset_similarity_service import ItemsetSimilarityService, SetSimConfig
from services.neo4j_sanchez_service import FrequentItemsetRow


class ReportingService:
    def __init__(self, ic_index: ICIndex, set_sim: ItemsetSimilarityService):
        self.idx = ic_index
        self.set_sim = set_sim

    def cohesion(self, X: List[FrequentItemsetRow], cfg: SetSimConfig) -> Dict:
        if not X:
            return {
                "n_itemsets": 0,
                "n_pairs": 0,
                "mean_similarity": 0.0,
                "pct_high_sim": 0.0,
                "pct_low_sim": 0.0
            }
            
        similarities = []
        n_pairs = 0
        
        # Compare all pairs
        for i in range(len(X)):
            for j in range(i+1, len(X)):
                a, b = X[i], X[j]
                sim = self.set_sim.set_similarity(a.items, b.items, cfg)
                similarities.append(sim)
                n_pairs += 1
                
        # Convert to numpy for efficient stats
        sims = pd.Series(similarities)
        
        return {
            "n_itemsets": len(X),
            "n_pairs": n_pairs,
            "mean_similarity": float(sims.mean()),
            "pct_high_sim": float((sims >= 0.8).mean()),
            "pct_low_sim": float((sims < 0.2).mean())
        }

    def summarize_mean(self, df: pd.DataFrame) -> float:
        return float(df["similarity"].mean()) if not df.empty else 0.0