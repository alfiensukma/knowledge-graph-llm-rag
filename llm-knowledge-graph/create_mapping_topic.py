import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from services.graph_service import GraphService
from services.topic_service import TopicExtractionService

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'ISSN:?\s*\d{4}-\d{4}', '', text)
    text = re.sub(r'www\.[^\s]+', '', text)
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def fetch_papers_from_graph(graph_service):
    query = """
    MATCH (p:Paper)
    WHERE p.filename IS NOT NULL AND p.filename <> ''
    WITH p.filename AS filename, collect(p) AS papers
    WITH filename, papers,
         any(q IN papers WHERE EXISTS { MATCH (q)-[:HAS_TOPIC]->(:Topic) }) AS processed
    WITH filename, papers, processed,
         [q IN papers WHERE processed AND EXISTS { MATCH (q)-[:HAS_TOPIC]->(:Topic) }][0] AS repWith
    WITH filename, processed, coalesce(repWith, papers[0]) AS rep
    RETURN elementId(rep) AS pid,
           filename,
           rep.title AS title,
           (CASE WHEN processed THEN 1 ELSE 0 END) AS topic_count
    ORDER BY filename
    """
    rows = graph_service.graph.query(query)
    papers = []
    for r in rows:
        papers.append({
            "pid": r["pid"],
            "filename": r.get("filename"),
            "title": r.get("title"),
            "topic_count": r.get("topic_count", 0)
        })
    return papers

def display_paper_status(papers):
    selectable = []
    print("\n" + "=" * 110)
    print("PAPER STATUS (PROCESSED = has HAS_TOPIC, select only NOT PROCESSED)")
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
        status = "(PROCESSED)" if p["topic_count"] > 0 else "(NOT PROCESSED)"
        if p["topic_count"] == 0:
            selectable.append((idx, p))
            color = "\033[93m"  # yellow
        else:
            color = "\033[92m"  # green
        reset = "\033[0m"
        print(f"{color}{idx:2d}. {filename:<70} {status:<15}{reset}")

    print("=" * 110)
    processed = sum(1 for p in papers if p["topic_count"] > 0)
    print(f"Summary: {len(papers)} total | {processed} processed | {len(selectable)} available")
    if selectable:
        nums = ", ".join(str(i) for i, _ in selectable)
        print(f" > Valid choices (NOT PROCESSED): {nums}")
    else:
        print(" > No unmapped papers.")
    return selectable

def select_paper(selectable, all_papers):
    if not selectable:
        return None
    valid_indices = {i: p for i, p in selectable}
    while True:
        choice = input("\nEnter paper number (or 'q' to quit): ").strip()
        if choice.lower() == 'q':
            return None
        if not choice.isdigit():
            print(" > Invalid input.")
            continue
        num = int(choice)
        if num in valid_indices:
            return valid_indices[num]
        if 1 <= num <= len(all_papers):
            if all_papers[num - 1]["topic_count"] > 0:
                print(" > Paper has already HAS_TOPIC. Choose another.")
            else:
                print(" > Number not in selectable range.")
        else:
            print(" > Out of range.")

def load_pdf_text_for_paper(paper, pdf_root="data/pdfs"):
    filename = paper["filename"]
    path = os.path.join(pdf_root, filename) if filename else None
    if not path or not os.path.isfile(path):
        print(f" > PDF file '{filename}' not found in {pdf_root}. Fallback to using title only.")
        title = paper.get("title") or ""
        return title.strip(), 0, False
    # Load PDF
    try:
        loader = PyPDFLoader(path)
        docs = loader.load()
        pages = []
        for d in docs:
            if d.page_content and isinstance(d.page_content, str):
                pages.append(clean_text(d.page_content))
        full_text = "\n".join(pages)
        if not full_text.strip():
            print(" > Empty PDF content after cleaning. Fallback to title.")
            return (paper.get("title") or "").strip(), len(pages), False
        return full_text, len(pages), True
    except Exception as e:
        print(f" > Fail to read PDF: {e}. Fallback to title.")
        return (paper.get("title") or "").strip(), 0, False

def main():
    load_dotenv()

    # Config
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

    topic_handler = TopicExtractionService(llm=llm, graph_service=graph_service)

    # Load PDF text
    print(f"\nLoading PDF for: {selected['filename']}")
    full_text, page_count, used_pdf = load_pdf_text_for_paper(selected)
    if not full_text.strip():
        print(" > No text to process. Abort.")
        return
    source_desc = f"{page_count} pages PDF" if used_pdf else "TITLE ONLY"
    print(f" > Context source: {source_desc} | length chars={len(full_text)}")

    paper_id = selected["pid"]
    print(f"\nMapping topics for Paper(id={paper_id}) ...")

    # Topic processing
    print("\nProcessing topics...")
    try:
        topics = topic_handler.get_validated_topics_for_text(full_text)
        print(f"Validated topics ({len(topics)}): {topics}")
        
        if topics:
            unique_topics = sorted(set(t for t in topics if isinstance(t, str) and t.strip()))
            if not unique_topics:
                print(" > No valid topic strings after normalization.")
            else:
                # Attempt linking only for topics that exist in DB
                link_result = graph_service.graph.query("""
                    MATCH (p:Paper {filename: $filename})
                    WITH p
                    UNWIND $topics AS topicLabel
                    OPTIONAL MATCH (t:Topic {label: topicLabel})
                    WITH p, topicLabel, t
                    CALL {
                      WITH p, t
                      WITH p, t
                      WHERE t IS NOT NULL
                      MERGE (p)-[:HAS_TOPIC]->(t)
                      RETURN count(*) AS _c
                    }
                    RETURN collect({topic: topicLabel, exists: t IS NOT NULL}) AS status
                """, {"filename": selected["filename"], "topics": unique_topics})

                status = link_result[0]["status"] if link_result else []
                missing = [s["topic"] for s in status if not s["exists"]]
                linked = [s["topic"] for s in status if s["exists"]]

                print(f" > Linked topics: {linked}")
                if missing:
                    print(f" > Skipped (no matching Topic node): {missing}")

                # Verification
                verify = graph_service.graph.query("""
                    MATCH (p:Paper {filename:$filename})-[:HAS_TOPIC]->(t:Topic)
                    RETURN count(t) AS total, collect(t.label) AS labels
                """, {"filename": selected["filename"]})
                if verify:
                    print(f" > Verification: {verify[0]['total']} HAS_TOPIC now -> {verify[0]['labels']}")
                else:
                    print(" > Verification query returned no rows.")
        else:
            print(" > No topics to link.")
    except Exception as e:
        print(f" > Error mapping topics: {e}")
        import traceback
        traceback.print_exc()

    print("\nDone.")

if __name__ == "__main__":
    main()
