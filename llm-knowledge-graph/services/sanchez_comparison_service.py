from __future__ import annotations
import os
import json
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


class SanchezComparisonService:

    def __init__(self, result_dir: str):
        self.result_dir = result_dir


    def load_results(self, filename: str) -> Optional[Dict]:
        path = os.path.join(self.result_dir, filename + ".json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None


    def extract_metrics(self, results: Dict) -> Dict:
        if not results:
            return {}
            
        return {
            "mean": results.get("mean_similarity", 0.0),
            "pct_high_sim": results.get("pct_high_sim", 0.0),
            "pct_low_sim": results.get("pct_low_sim", 0.0),
            "cluster_score": results.get("pct_high_sim", 0.0) / max(0.01, results.get("pct_low_sim", 1.0)),
            "variance_score": results.get("std", 0.0) ** 2 if "std" in results else 0.0
        }


    def compare_methods(self, method_files: Dict[str, str]) -> Dict:
        results = {}
        all_metrics = {}

        for method, filename in method_files.items():
            results = self.load_results(filename)
            if results is not None:
                metrics = self.extract_metrics(results)
                all_metrics[method] = metrics

        if not all_metrics:
            return {"error": "No valid results loaded"}

        dist_comparison = pd.DataFrame(all_metrics).round(3)
        patterns = {
            "highest_similarity": max(all_metrics.items(), key=lambda x: x[1]["mean"])[0],
            "most_clustered": max(all_metrics.items(), key=lambda x: x[1]["cluster_score"])[0],
            "most_diverse": min(all_metrics.items(), key=lambda x: x[1]["variance_score"])[0]
        }

        quality_ranking = sorted(
            all_metrics.keys(),
            key=lambda m: (
                all_metrics[m]["mean"] * 0.4 +  # weight mean similarity
                all_metrics[m]["cluster_score"] * 0.4 +  # weight clustering
                (1.0 - all_metrics[m]["variance_score"]) * 0.2  # weight (inverse) variance
            ),
            reverse=True
        )

        results = {
            "metrics_comparison": dist_comparison.to_dict(),
            "patterns": patterns,
            "quality_ranking": quality_ranking,
            "interpretation": {
                "best_method": quality_ranking[0],
                "reasoning": f"{quality_ranking[0]} achieves the best balance of "
                           f"similarity (cohesive topics), clustering (distinct groups), "
                           f"and controlled variance (clear structure)."
            }
        }

        return results


    def print_comparison_report(self, comparison_results: Dict):
        if "error" in comparison_results:
            print(f"Error: {comparison_results['error']}")
            return

        print("\n=== Sanchez Similarity Comparison Report ===\n")
        
        metrics_df = pd.DataFrame(comparison_results["metrics_comparison"])
        print("Method Comparison Metrics:")
        print(metrics_df.round(3))
        print()
        
        patterns = comparison_results["patterns"]
        print("Key Patterns:")
        print(f"- Highest average similarity: {patterns['highest_similarity']}")
        print(f"- Most distinct topic clusters: {patterns['most_clustered']}")
        print(f"- Most topic diversity: {patterns['most_diverse']}")
        print()

        print()

        print("Overall Quality Ranking:")
        for i, method in enumerate(comparison_results["quality_ranking"], 1):
            print(f"{i}. {method}")
        print()

        interp = comparison_results["interpretation"]
        print("Interpretation:")
        print(f"Best performing method: {interp['best_method']}")
        print(f"Reasoning: {interp['reasoning']}")