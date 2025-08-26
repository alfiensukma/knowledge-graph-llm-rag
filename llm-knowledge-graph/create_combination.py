import os, traceback
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from services.graph_service import GraphService
from services.llm_combination_service import LLMCombinationService

def fetch_papers_from_graph(graph_service):
    query = """
    MATCH (p:Paper)
    WHERE p.filename IS NOT NULL AND p.filename <> ''
    WITH p.filename AS filename, collect(p) AS papers
    WITH filename, papers,
         any(q IN papers WHERE EXISTS { MATCH (q)-[:HAS_TOPIC_COMBINATION]->(:TopicCombination) }) AS has_combinations
    WITH filename, papers, has_combinations,
         [q IN papers WHERE has_combinations AND EXISTS { MATCH (q)-[:HAS_TOPIC_COMBINATION]->(:TopicCombination) }][0] AS repWith
    WITH filename, has_combinations, coalesce(repWith, papers[0]) AS rep
    RETURN rep.id AS paper_id,
           elementId(rep) AS pid,
           filename,
           rep.title AS title,
           (CASE WHEN has_combinations THEN 1 ELSE 0 END) AS combination_count
    ORDER BY filename
    """
    rows = graph_service.graph.query(query)
    papers = []
    for r in rows:
        papers.append({
            "paper_id": r["paper_id"],
            "pid": r["pid"],
            "filename": r.get("filename"),
            "title": r.get("title"),
            "combination_count": r.get("combination_count", 0)
        })
    return papers

def display_paper_status(papers):
    selectable = []
    print("\n" + "=" * 110)
    print("PAPER STATUS (PROCESSED = has HAS_TOPIC_COMBINATION and node TopicCombination)")
    print("=" * 110)
    if not papers:
        print("No Paper nodes found in database.")
        return []

    for idx, p in enumerate(papers, 1):
        filename = p.get("filename") or "(no filename)"
        title = (p.get("title") or "").strip()
        if len(title) > 90:
            title_disp = title[:87] + "..."
        else:
            title_disp = title
        
        has_combinations = p["combination_count"] > 0
        status = "(PROCESSED)" if has_combinations else "(UNPROCESSED)"
        
        if has_combinations:
            color = "\033[92m"
        else:
            color = "\033[93m"
            selectable.append((idx, p))
            
        reset = "\033[0m"
        print(f"{color}{idx:2d}. {filename:<70} {status:<15}{reset}")

    print("=" * 110)
    processed = sum(1 for p in papers if p["combination_count"] > 0)
    print(f"Summary: {len(papers)} total | {processed} processed | {len(selectable)} available for combination")
    if selectable:
        nums = ", ".join(str(i) for i, _ in selectable)
        print(f" > Valid choices (UNPROCESSED papers): {nums}")
    else:
        print(" > No unprocessed papers found.")
    return selectable

def select_paper(selectable, all_papers):
    if not selectable:
        print(" > No unprocessed papers available for combination generation.")
        return None
    
    valid_choices = {i: p for i, p in selectable}
    valid_numbers = sorted(valid_choices.keys())
    
    print(f"\n > Available choices: {', '.join(map(str, valid_numbers))}")
    
    while True:
        try:
            choice = input("\nEnter paper number (or 'q' to quit): ").strip()
            
            if choice.lower() == 'q':
                print(" > Exiting...")
                return None
            
            if not choice.isdigit():
                print(" > Invalid input! Please enter a number or 'q' to quit")
                continue
            
            choice_num = int(choice)
            
            if choice_num in valid_choices:
                selected_paper = valid_choices[choice_num]
                print(f"\nSelected: {selected_paper['filename']}")
                return selected_paper
            
            elif 1 <= choice_num <= len(all_papers):
                paper = all_papers[choice_num - 1]
                if paper["combination_count"] > 0:
                    print(f" > Paper '{paper['filename']}' already has combinations!")
                    print(f" > Please select from available choices: {', '.join(map(str, valid_numbers))}")
                else:
                    print(f" > Invalid choice! Available choices: {', '.join(map(str, valid_numbers))}")
            else:
                print(f" > Invalid choice! Please enter a number from: {', '.join(map(str, valid_numbers))}")
                
        except ValueError:
            print(" > Invalid input! Please enter a number or 'q' to quit")
        except KeyboardInterrupt:
            print("\n Interrupted by user. Exiting...")
            return None

def main():
    load_dotenv()
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

    print("Initializing services...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY, temperature=0)
    graph_service = GraphService(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

    # Fetch Paper nodes
    papers = fetch_papers_from_graph(graph_service)
    selectable = display_paper_status(papers)
    selected = select_paper(selectable, papers)
    if not selected:
        print(" > No selection. Exit.")
        return

    svc = LLMCombinationService(llm=llm, graph_service=graph_service)

    paper_id = selected["paper_id"]
    filename = selected["filename"]
    print(f"\nGenerating combinations for: {filename}")
    print(f"Paper ID: {paper_id}")

    # Generate combinations for the selected paper (max_k=5)
    print("\nGenerating topic combinations...")
    try:
        result = svc.generate_combinations_for_paper(paper_id, max_k=5, repair_missing=True)
        if result:
            print(f"\nSuccessfully generated {len(result)} combinations for paper: {filename}")

            topics_count = len(svc._fetch_topics_for_paper(paper_id))
            expected_total = sum(1 for r in range(1, min(topics_count, 5) + 1) 
                            for _ in __import__('itertools').combinations(range(topics_count), r))
            print(f"Expected combinations (max_k=5): {expected_total}")
            print(f"Actual generated: {len(result)}")
            if len(result) < expected_total:
                print(f"Missing: {expected_total - len(result)} combinations")
        else:
            print(f"\nNo combinations generated for paper: {filename}")
    except Exception as e:
        print(f"\nError generating combinations: {e}")
        traceback.print_exc()

    print("\nDone.")

if __name__ == "__main__":
    main()
