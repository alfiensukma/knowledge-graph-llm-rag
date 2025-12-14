import os
from services.sanchez_comparison_service import SanchezComparisonService

if __name__ == "__main__":
    result_dir = os.path.join(os.path.dirname(__file__), "data", "sanchez-result")
    comparison = SanchezComparisonService(result_dir)
    
    method_files = {
        "LLM": input("Enter LLM result filename (without extension): "),
        "LSA": input("Enter LSA result filename (without extension): "),
        "LDA": input("Enter LDA result filename (without extension): ")
    }

    results = comparison.compare_methods(method_files)
    comparison.print_comparison_report(results)