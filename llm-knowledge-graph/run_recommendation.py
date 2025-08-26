import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from services.graph_service import GraphService
from services.recommendation_service import RecommendationService

def fetch_papers_from_graph(graph_service):
    query = """
    MATCH (p:Paper)
    WHERE p.filename IS NOT NULL AND p.filename <> ''
    WITH p.filename AS filename, collect(p) AS papers
    WITH filename, papers[0] AS rep
    RETURN elementId(rep) AS pid,
           filename,
           rep.title AS title
    ORDER BY filename
    """
    rows = graph_service.graph.query(query)
    papers = []
    for r in rows:
        papers.append({
            "pid": r["pid"],
            "filename": r.get("filename"),
            "title": r.get("title")
        })
    return papers

def display_papers(papers):
    print("\n" + "=" * 110)
    print("PAPER LIST (select papers for recommendation sample)")
    print("=" * 110)
    if not papers:
        print("No Paper nodes found in database.")
        return []

    for idx, p in enumerate(papers, 1):
        filename = p.get("filename") or "(no filename)"
        title = (p.get("title") or "").strip()
        if len(title) > 70:
            title_disp = title[:67] + "..."
        else:
            title_disp = title
        print(f"{idx:2d}. {filename:<50} | {title_disp}")

    print("=" * 110)
    print(f"Total: {len(papers)} papers available")
    return papers

def select_papers(papers):
    if not papers:
        print(" > No papers available for selection.")
        return []
    
    print(f"\n > Available choices: 1-{len(papers)}")
    
    while True:
        choice = input("\nEnter paper numbers separated by commas (e.g., 1,2,3) or 'q' to quit: ").strip()
        if choice.lower() == 'q':
            return []
        
        try:
            # Parse the input
            selected_indices = []
            for token in choice.split(','):
                token = token.strip()
                if token.isdigit():
                    num = int(token)
                    if 1 <= num <= len(papers):
                        selected_indices.append(num - 1)
                    else:
                        print(f" > Number {num} is out of range (1-{len(papers)})")
                        break
                elif token:
                    print(f" > Invalid input: '{token}'. Please use numbers only.")
                    break
            else:
                if selected_indices:
                    selected_papers = [papers[i] for i in selected_indices]
                    selected_names = [papers[i]['filename'] for i in selected_indices]
                    print(f"\nSelected papers: {', '.join(selected_names)}")
                    return selected_papers
                else:
                    print(" > No valid numbers entered.")
        except Exception as e:
            print(f" > Error parsing input: {e}")

def main():
    load_dotenv()
    
    # Konfigurasi
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    
    print("Initializing services...")
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY, temperature=0)
    graph_service = GraphService(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
    recommendation_service = RecommendationService(llm=llm, graph_service=graph_service)
    
    # Fetch and display papers
    papers = fetch_papers_from_graph(graph_service)
    if not papers:
        print("No papers found in the database.")
        return
    
    display_papers(papers)
    selected_papers = select_papers(papers)
    
    if not selected_papers:
        print(" > No papers selected. Exiting.")
        return
    
    # Extract paper IDs for recommendation
    user_paper_ids = [paper["pid"] for paper in selected_papers]
    selected_names = [paper["filename"] for paper in selected_papers]
    
    print(f"\n--- Generating Recommendations for Selected Papers ---")
    print(f" > Selected: {', '.join(selected_names)}")
    print(f" > Paper IDs: {user_paper_ids}")
    
    recommendations = recommendation_service.get_llm_recommendations(user_paper_ids)
    
    print("\n--- LLM Recommendations ---")
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"{i:2d}. {rec['filename']}")
            print(f"    Title: {rec['title']}")
            print(f"    Topics: {rec['topics']}")
            print()
    else:
        print("  > No recommendations generated.")
    
    print("Recommendations generated successfully!")

if __name__ == "__main__":
    main()