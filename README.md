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
- **Python** 3.10+  
- **Neo4j** (local or cloud) running and reachable
- **Google Gemini API Key** (from [Google AI Studio](https://aistudio.google.com/)) - i'm using free version, 1.000.000 tokens/minutes
- **PDF files** placed under `llm-knowledge-graph/data/pdfs`

Optional (only if you run native Cypher/GDS Apriori variants):
- **APOC** & **GDS** plugins in Neo4j

---

## Setup & Installation

```bash
# 1) Create virtual environment
python -m venv venv

# 2) Activate venv
# Windows (Command Prompt):
.venv\Scripts\activate

# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# macOS/Linux:
source .venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Go to project root
cd llm-knowledge-graph
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

Put your .env file here:

```
knowledge-graph-llm-rag
└─ venv
└─ .env
└─ .gitiginore
└─ llm-knowledge-graph/
└─ requirements.txt
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

#### Prerequisites

Ensure you're in the correct directory and venv is active:
<img width="753" height="32" alt="Screenshot 2025-08-25 080019" src="https://github.com/user-attachments/assets/a3c10e22-1668-4954-a073-36627bad0282" />

#### Run this command

```bash
python create_topic_from_cso.py
```

#### Expected Result
<img width="993" height="415" alt="result cso_topic" src="https://github.com/user-attachments/assets/3603102d-686d-41a9-b4d5-7d51ebd976aa" />

> This process will feel long when retrieving data from an RDF file for the first time.

---

### 2) Create Paper

Creates `(:Paper)` from PDFs.

#### Prerequisites

Ensure you're in the correct directory and venv is active:
<img width="753" height="32" alt="Screenshot 2025-08-25 080019" src="https://github.com/user-attachments/assets/94912743-ad78-4b0d-af78-e3adc70a2b47" />

#### Make sure there is a PDF file

- Please note: You can only process one PDF file at a time. This project uses a limited tokens (version Gemini).
- You can select an UNPROCESSED PDF file from list to generate into a node.
- To avoid being warned that the token usage limit has been reached, please pause for at least 1 minute after finishing processing 1 PDF file.

#### Run this command

```bash
python create_paper.py
```

#### Expected Result
<img width="1021" height="780" alt="image" src="https://github.com/user-attachments/assets/a824a17c-fafb-4c07-b17c-9c18a26759e8" />

Graph:
<img width="982" height="466" alt="result_create_paper_graph" src="https://github.com/user-attachments/assets/08dfa20e-6d95-41d1-8a7f-d24ca34a424e" />

---

### 3) Topic Mapping using LLMs

Links topics via LLM (validated against CSO) → `(:Paper)-[:HAS_TOPIC]->(:Topic)`.

#### Prerequisites

Ensure you're in the correct directory and venv is active:
<img width="753" height="32" alt="Screenshot 2025-08-25 080019" src="https://github.com/user-attachments/assets/c4c18767-fac9-484f-91f5-44ff5d561d35" />

#### Make sure there is a Paper node and Topic node

- Please note: You can ONLY process ONE selected Paper at a time. This project uses a limited tokens (version Gemini).
- You can select an UNPROCESSED Paper from list to generate topic mapping.
- The list of papers that appears only shows PDFs that have been generated as nodes. Therefore, repeat step 2 to generate a new paper.

#### Run this command

```bash
python create_mapping_topic.py
```

#### Expected Result
<img width="1037" height="641" alt="result_topic_mapping" src="https://github.com/user-attachments/assets/c99e28d9-d5ec-459f-9b18-02f1a0aea85a" />

Graph:
<img width="1065" height="524" alt="result topic_mapping" src="https://github.com/user-attachments/assets/65cdd54f-7315-4a9c-9dcb-8bdb85659525" />

---

### 4) Topic Modeling (LDA or LSA) using LLMs

Runs topic modeling **LDA** or **LSA** on your PDFs and prints results using LLM.

#### Prerequisites

Ensure you're in the correct directory and venv is active:
<img width="753" height="32" alt="Screenshot 2025-08-25 080019" src="https://github.com/user-attachments/assets/09029bcf-d1a1-4dbf-8eac-6b9b012a812a" />

#### Run this command

```bash
python run_llm_topic_modeling.py
```

#### Expected Result
<img width="951" height="893" alt="result_llm_modeling" src="https://github.com/user-attachments/assets/fcc67604-ffc0-4f0c-ac00-38ea368ab78d" />

---

### 5) Topic Modeling (LDA or LSA)

Runs classical topic modeling **LDA** or **LSA** (scikit-learn) on your PDFs and prints results.

#### Prerequisites

Ensure you're in the correct directory and venv is active:
<img width="753" height="32" alt="Screenshot 2025-08-25 080019" src="https://github.com/user-attachments/assets/a9a5d524-1ef8-4fd7-82a4-225ecbfe25aa" />

#### Run this command

```bash
python run_topic_modeling.py
```

#### Expected Result
<img width="396" height="794" alt="result_lsa" src="https://github.com/user-attachments/assets/3b169d12-3559-41d4-b6b2-fd33eca7d37d" />
<img width="440" height="581" alt="result_lda" src="https://github.com/user-attachments/assets/673f02e4-3590-4aae-9c72-6e4a54df7e31" />

---

### 6) Topic Combinations using LLMs

Generate topic combinations per paper → creates `(:TopicCombination)` and `(:Paper)-[:HAS_TOPIC_COMBINATION]->(:TopicCombination)`:

#### Prerequisites

Ensure you're in the correct directory and venv is active:
<img width="753" height="32" alt="Screenshot 2025-08-25 080019" src="https://github.com/user-attachments/assets/7089d4fc-4162-4e55-a052-bc7e399d2045" />

#### Make sure there is a Paper node and Topic node

- Please note: You can ONLY process ONE selected Paper at a time. This project uses a limited tokens (version Gemini).
- Make sure all required PDF files have been generated into nodes

#### Run this command

```bash
python create_combination.py
```

#### Expected Result
<img width="987" height="864" alt="result_combination" src="https://github.com/user-attachments/assets/ce629449-7946-4c77-9b5d-10584098c457" />

Graph:
<img width="519" height="465" alt="result_combination_graph" src="https://github.com/user-attachments/assets/47febfaa-ecff-4df1-bd88-4df77e34bfbd" />

---

### 7) Apriori-like mining via LLM

Run Apriori-like mining via LLM → creates `(:FrequentTopicSet)`, `(:LeftTopicSet)-[:RULES]->( :RightTopicSet)` with `support` & `confidence`:

#### Prerequisites

Ensure you're in the correct directory and venv is active:
<img width="753" height="32" alt="Screenshot 2025-08-25 080019" src="https://github.com/user-attachments/assets/2ae77607-3553-4fff-ae40-d4828fac2fb4" />

#### Make sure there is a Paper node and Topic node

Make sure all required PDF files have been generated into nodes

#### Run this command

```bash
python run_llm_apriori.py
```

> The Apriori logic (frequent itemsets, rules) is **driven by LLM**; Cypher is used only to persist the results into Neo4j.

#### Expected Result
<img width="940" height="721" alt="result_llm_apriori" src="https://github.com/user-attachments/assets/eeacbf05-a088-4652-8788-8ec1584695df" />

Graph:
<img width="948" height="445" alt="result_llm_apriori_graph" src="https://github.com/user-attachments/assets/b02069bb-9f92-46f5-9069-46f9854eb773" />

---

### 8) Recommendations (LLM Apriori-like)

Recommends papers based on the learned co-occurrence patterns:

#### Prerequisites

Ensure you're in the correct directory and venv is active:
<img width="753" height="32" alt="Screenshot 2025-08-25 080019" src="https://github.com/user-attachments/assets/682a5dde-3166-4304-90fa-9e58c7a745c6" />

#### Make sure there is a Paper node and Topic node

- The list of papers that appears only shows PDFs that have been generated as nodes. Therefore, repeat step 2 to generate a new paper.
- You can select more than 1 paper to be used as a sample. For example, `Select Paper: 1, 2, 3`

#### Run this command

```bash
python run_recommendation.py
```

#### Expected Result
<img width="1091" height="741" alt="result_recommendation" src="https://github.com/user-attachments/assets/2fdceb43-9275-4363-a45b-c96d5a7555fb" />

---

### 9) Chatbot (Streamlit)

Run embedding first

```bash
python services\embedding_service.py
```
<img width="1051" height="155" alt="image" src="https://github.com/user-attachments/assets/3c7f07a5-5917-46f1-a336-f6248c0a6e2f" />

Graph + vector:
<img width="1003" height="442" alt="image" src="https://github.com/user-attachments/assets/d3e11fd8-c0db-4495-9431-28975aac4c2a" />

> To avoid the warning that the token usage limit has been reached, please ONLY use ONE pdf file during the embedding process.

Browse & query your graph via a simple UI:

```bash
cd chatbot
streamlit run main.py
```

Open your browser at `http://localhost:8501`.

#### Expected Result
![chatbot](https://github.com/user-attachments/assets/9c2d2935-ac4f-4110-aaa1-e9e89aebe8e5)

---

## Script Index

| Script | Purpose |
|---|---|
| `create_topic_from_cso.py` | Imports **CSO** topics + hierarchy into Neo4j. |
| `create_paper.py` | Creates `(:Paper)` nodes from PDFs. |
| `create_mapping_topic.py` | Links topics (`HAS_TOPIC`) using LLM. |
| `run_topic_modeling.py` | Runs **LDA** or **LSA** (sklearn) and prints topics/terms. |
| `run_llm_topic_modeling.py` | Runs **LDA** or **LSA** using LLMs and prints topics/terms (LSA/LDA-like). |
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
