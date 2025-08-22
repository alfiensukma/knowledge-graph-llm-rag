# Knowledge Graph Construction with LLMs

Build a **Computer Science** knowledge graph from PDFs using **LLMs**, store it in **Neo4j**, validate topics against the **Computer Science Ontology (CSO)**, and explore via a **Streamlit chatbot**. The project also includes **LLM Apriori-like mining** for associations and optional **LDA/LSA topic modeling** pipelines whose outputs are mapped back to CSO topics.

## Contents
- [Prerequisites](#prerequisites)
- [Setup & Installation](#setup--installation)
- [Environment Variables](#environment-variables)
- [Data](#data)
- [How to Run](#how-to-run)
  - [Create Topics from CSO (RDF)](#1-create-topics-from-cso-rdf)
  - [Create Paper + Topics using LLM](#2-create-paper--topics-using-llm)
  - [Switch Topic Extraction Mode (LLM direct vs LDA/LSA-like LLM)](#switch-topic-extraction-mode-llm-direct-vs-ldalsa-like-llm)
  - [Topic Modeling (LDA or LSA)](#3-topic-modeling-lda-or-lsa)
  - [Create Paper Nodes Only (no topics)](#4-create-paper-nodes-only-no-topics)
  - [Apriori (Topic Combinations & LLM Apriori-like)](#5-apriori-topic-combinations--llm-apriori-like)
  - [Recommendations (from LLM Apriori-like)](#6-recommendations-from-llm-apriori-like)
  - [Chatbot (Streamlit)](#7-chatbot-streamlit)
- [Script Index](#script-index)
- [Neo4j Notes](#neo4j-notes)
- [Troubleshooting](#troubleshooting)
- [Acknowledgments](#acknowledgments)

---

## Prerequisites
- **Python** 3.8+  
- **Neo4j** (local or cloud) running and reachable
- **Google Gemini API Key** (from [Google AI Studio](https://aistudio.google.com/))
- **PDF files** placed under `llm-knowledge-graph/data/pdfs`

Optional (only if you run native Cypher/GDS Apriori variants):
- **APOC** & **GDS** plugins in Neo4j

---

## Setup & Installation

```bash
# 1) Go to project root
cd llm-knowledge-graph

# 2) Create virtual environment
python -m venv venv

# 3) Activate venv
# Windows (PowerShell):
.venv\Scripts\Activate
# macOS/Linux:
# source venv/bin/activate

# 4) Install dependencies
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` in the project root:

```env
# Gemini
GEMINI_API_KEY=your_api_key_from_google_ai_studio

# Neo4j
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

> **Tip:** Ensure your Neo4j instance is running and accepts Bolt connections at the URI you set.

---

## Data

Put your CS papers (PDF) here:

```
llm-knowledge-graph/
└─ data/
   └─ pdfs/
      ├─ paper1.pdf
      ├─ paper2.pdf
      └─ ...
```

---

## How to Run

### 1) Create Topics from CSO (RDF)

This parses the **Computer Science Ontology** (RDF) and loads `(:Topic)` nodes (and their hierarchy) into Neo4j.

```bash
python create_topic_from_cso.py
```

---

### 2) Create Paper + Topics using LLM

End-to-end: creates `(:Paper)` from PDFs AND links topics via LLM (validated against CSO) → `(:Paper)-[:HAS_TOPIC]->(:Topic)`.

```bash
python create_paper_with_hasTopic.py
```

#### Switch Topic Extraction Mode (LLM direct vs LDA/LSA-like LLM)

Open `main.py` and set exactly one of these:

```python
USE_TOPIC_SERVICE = False        # direct LLM topic extraction/validation
USE_LLM_TOPIC_MODELING = True    # LLM that emulates LSA/LDA then maps to CSO
```

- `TopicExtractionService` (direct LLM) uses the **entire document content** to extract up to 10 topics and validates them against CSO.
- `LLMTopicModelingService` runs **LSA-like & LDA-like (simulated by LLM)** over the **full document text**, then maps top terms back to CSO topics and links them.

> Both modes ultimately create `HAS_TOPIC` relationships to **existing CSO topics** (no new topic nodes).

---

### 3) Topic Modeling (LDA or LSA)

Runs classical **LDA** or **LSA** (scikit-learn) on your PDFs, prints results, and maps top terms to CSO using LLM.

```bash
python run_topic_modeling.py
```

Choose **one** model by setting flags inside `run_topic_modeling.py`:

```python
RUN_LSA = True
RUN_LDA = False
```

> The script prevents both being `True` at the same time.

---

### 4) Create Paper Nodes Only (no topics)

If you want paper nodes without linking topics:

```bash
python create_paper.py
```

This only creates `(:Paper)` nodes from PDFs.

---

### 5) Apriori (Topic Combinations & LLM Apriori-like)

**(a) Generate topic combinations per paper** → creates `(:TopicCombination)` and `(:Paper)-[:HAS_TOPIC_COMBINATION]->(:TopicCombination)`:

```bash
python create_combination.py
```

**(b) Run Apriori-like mining via LLM** → creates `(:FrequentTopicSet)`, `(:LeftTopicSet)-[:RULES]->( :RightTopicSet)` with `support` & `confidence`:

```bash
python run_llm_apriori.py
```

> The Apriori logic (frequent itemsets, rules) is **driven by LLM**; Cypher is used only to persist the results into Neo4j.

---

### 6) Recommendations (from LLM Apriori-like)

Recommends papers based on the learned co-occurrence patterns:

```bash
python run_recommendation.py
```

---

### 7) Chatbot (Streamlit)

Browse & query your graph via a simple UI:

```bash
cd chatbot
streamlit run main.py
```

Open your browser at `http://localhost:8501`.

---

## Script Index

| Script | Purpose |
|---|---|
| `create_topic_from_cso.py` | Imports **CSO** topics + hierarchy into Neo4j. |
| `create_paper.py` | Creates `(:Paper)` nodes from PDFs (no topics). |
| `create_paper_with_hasTopic.py` | Creates papers AND links topics (`HAS_TOPIC`) using LLM (direct or LSA/LDA-like). |
| `run_topic_modeling.py` | Runs **LDA** or **LSA** (sklearn), prints topics/terms, maps to CSO, and links `HAS_TOPIC`. |
| `create_combination.py` | Generates all topic combinations per paper and persists `(:TopicCombination)`. |
| `run_llm_apriori.py` | Runs **LLM Apriori-like** to create `(:FrequentTopicSet)` and association rules. |
| `run_recommendation.py` | Recommends papers using the LLM Apriori-like outputs. |
| `chatbot/main.py` | Streamlit chatbot for querying and recommending papers. |

---

## Neo4j Notes

- This project uses the official Bolt driver through a small `GraphService` wrapper.
- If you later want to run **pure Cypher/GDS Apriori** (instead of LLM Apriori-like), you’ll need **APOC** and **GDS** plugins enabled and configured.  
- For full-text topic lookups, the code creates a **FULLTEXT INDEX** on `(:Topic {label})` automatically (if not present).
- Array property queries: if a node property is an array, compare using array syntax, e.g.  
  ```cypher
  MATCH (c:TopicCombination) 
  WHERE c.items = ['neural network'] 
  RETURN c
  ```

---

## Troubleshooting

- **No topics created / “No topics found”**  
  Ensure you ran `create_topic_from_cso.py` successfully and your Neo4j credentials are correct.

- **LLM prompt errors (missing template variables / dict has no attribute …)**  
  These happen when the LLM returns malformed JSON or the prompt placeholders aren’t escaped. The code includes guards and parsers; if it still happens, check console logs for the printed raw snippet.

- **Only a few topics get mapped**  
  That’s expected: mapping uses strict, CSO-guarded matching and confidence thresholds to avoid wrong links. Tune:
  - `min_confidence` (e.g., `0.85`)
  - `top_k_map_each` (terms per model)
  - `max_topics_in_prompt` (candidate pool size)

- **Long PDFs**  
  The LLM pipelines read **full document content**. If you hit model context limits, set a safety cap (e.g., `max_context_chars`) in `LLMTopicModelingService` initialization.

---

## Acknowledgments

- [Neo4j GraphAcademy](https://graphacademy.neo4j.com/) — Courses:  
  - [Constructing Knowledge Graphs with LLMs](https://graphacademy.neo4j.com/courses/llm-knowledge-graphs-construction)  
  - [Generative AI Workshop](https://graphacademy.neo4j.com/courses/genai-workshop)
- [Computer Science Ontology (CSO)](https://cso.kmi.open.ac.uk/topics/computer_science) — for topic validation
