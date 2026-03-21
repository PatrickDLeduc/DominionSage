# DominionSage — Step-by-Step Build Walkthrough

> A hybrid RAG system for Dominion card game intelligence.  
> Combines a structured card database with vector-searched rulebook knowledge to answer natural-language questions about cards, rules, combos, and strategy.

---

## How to Use This Document

Work through the phases **in order** — each one builds on the last. Every phase starts with what you're building, why it matters, and ends with a clear "done" checkpoint. Don't move on until you hit the checkpoint.

**Estimated total build time:** 12–18 hours across 6 phases.  
**Estimated total cost:** < $3.00 (API calls + free tiers).

---

## Prerequisites

Before you start Phase 1, make sure you have:

- [ ] Python 3.11+ installed
- [ ] An OpenAI API key (for embeddings and GPT-4o-mini)
- [ ] A free Supabase account created at [supabase.com](https://supabase.com)
- [ ] A GitHub repo initialized for the project
- [ ] These Python packages available: `supabase`, `openai`, `pdfplumber`, `streamlit`, `tiktoken`

```bash
pip install supabase openai pdfplumber streamlit tiktoken
```

---

## Project File Structure (Target)

Create files as you go — this is what the finished project looks like:

```
dominionsage/
├── app/
│   └── main.py                  # Streamlit UI (Phase 4)
├── data/
│   ├── scrape_cards.py          # Card data scraping (Phase 1)
│   ├── load_cards.py            # Card loading into Supabase (Phase 1)
│   ├── chunk_rulebooks.py       # PDF extraction + chunking (Phase 2)
│   └── embed_and_load.py        # Embedding + pgvector loading (Phase 2)
├── retrieval/
│   ├── router.py                # Query type classification (Phase 3)
│   ├── card_lookup.py           # Structured card queries (Phase 3)
│   ├── rules_search.py          # Vector similarity search (Phase 3)
│   ├── orchestrator.py          # Multi-path retrieval coordinator (Phase 3)
│   └── synthesizer.py           # LLM answer generation (Phase 3)
├── evals/
│   ├── questions.json           # 20 labeled eval questions (Phase 5)
│   ├── run_evals.py             # Automated eval runner (Phase 5)
│   └── results.csv              # Eval output (Phase 5)
├── requirements.txt
├── .env.example
└── README.md                    # Project documentation (Phase 6)
```

---

## Phase 1: Data Modeling & Card Ingestion

**⏱ Time:** 2–3 hours  
**🎯 Skill focus:** Structured data modeling, Supabase setup, Python scripting  
**📦 Deliverable:** A populated `cards` table with all Base + Seaside expansion cards

### Why This Phase Matters

> Not all AI context needs to be embedded. Structured data in a relational database is often faster, cheaper, and more precise than vector search for lookups with known attributes. This is the same principle as choosing between a hash join and a full table scan — the optimizer (your router, built in Phase 3) picks the best strategy.

### Steps

#### 1.1 — Create your Supabase project

- Go to [supabase.com](https://supabase.com) and create a new project (free tier is fine)
- Note your **project URL** and **anon key** — you'll need both
- Create a `.env` file locally (and `.env.example` for the repo):

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
OPENAI_API_KEY=sk-your-key
```

#### 1.2 — Enable pgvector

Open the Supabase SQL editor and run:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

You won't use vectors until Phase 2, but enabling it now means your schema is complete.

#### 1.3 — Create the cards table

Run this in the SQL editor:

```sql
CREATE TABLE cards (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    cost INTEGER,
    type TEXT,
    expansion TEXT NOT NULL,
    text TEXT,
    plus_actions INTEGER DEFAULT 0,
    plus_cards INTEGER DEFAULT 0,
    plus_buys INTEGER DEFAULT 0,
    plus_coins INTEGER DEFAULT 0
);
```

**Why denormalize +Actions/+Cards/+Buys/+Coins?** You could parse these from the card text at query time, but pre-extracting them into columns lets you write fast SQL filters like "Show me all cards that give +2 Actions" without needing an LLM to interpret card text. This is the classic data engineering tradeoff: **compute at ingest time to save compute at query time.**

#### 1.4 — Create the rulebook_chunks table (empty for now)

```sql
CREATE TABLE rulebook_chunks (
    id SERIAL PRIMARY KEY,
    expansion TEXT NOT NULL,
    source_page INTEGER,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(1536)
);
```

#### 1.5 — Write `data/scrape_cards.py`

Scrape card data from the [Dominion Wiki](http://wiki.dominionstrategy.com/) or use a community JSON dataset. You need: name, cost, type, expansion, full card text.

Tips:
- Start with **Base** and **Seaside** only (~60 cards total)
- Parse the +Actions/+Cards/+Buys/+Coins values from the card text during scraping
- Output to a local JSON file first so you can inspect before loading

#### 1.6 — Write `data/load_cards.py`

Read the scraped JSON and INSERT into Supabase:

```python
from supabase import create_client
import os, json

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

with open("data/cards.json") as f:
    cards = json.load(f)

for card in cards:
    supabase.table("cards").upsert(card).execute()
```

#### 1.7 — Validate with test queries

Run these 5 queries in the Supabase SQL editor (or via Python) and verify correct results:

```sql
-- By name
SELECT * FROM cards WHERE name = 'Chapel';

-- By cost
SELECT * FROM cards WHERE cost <= 3;

-- By expansion
SELECT * FROM cards WHERE expansion = 'Seaside';

-- By type
SELECT * FROM cards WHERE type ILIKE '%Action%';

-- By attribute
SELECT * FROM cards WHERE plus_actions >= 2;
```

### ✅ Phase 1 Checkpoint

- [ ] `cards` table exists in Supabase with pgvector enabled
- [ ] `rulebook_chunks` table exists (empty)
- [ ] All Base + Seaside cards loaded (~60 rows)
- [ ] All 5 validation queries return correct results
- [ ] `scrape_cards.py` and `load_cards.py` committed to repo

---

## Phase 2: Rulebook Chunking & Embedding

**⏱ Time:** 2–3 hours  
**🎯 Skill focus:** PDF text extraction, chunking strategies, vector embeddings  
**📦 Deliverable:** A populated `rulebook_chunks` table with embeddings for Base + Seaside rulebooks

### Why This Phase Matters

> An embedding converts text into a high-dimensional vector that captures **meaning**, not just keywords. Two sentences with zero shared words can have high cosine similarity if they mean similar things. Think of it like this: a traditional search index is a phone book (you need the exact name). A vector index is like describing someone's personality and finding people who are similar — even if they have completely different names.

### Steps

#### 2.1 — Download rulebook PDFs

Get the Base and Seaside rulebook PDFs (freely available from Rio Grande Games or the Dominion wiki). Save them to a `data/rulebooks/` directory.

#### 2.2 — Write `data/chunk_rulebooks.py`

```python
import pdfplumber
import tiktoken

enc = tiktoken.encoding_for_model("text-embedding-3-small")

def extract_text(pdf_path):
    """Extract all text from a PDF."""
    with pdfplumber.open(pdf_path) as pdf:
        pages = []
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages.append({"page": i + 1, "text": text})
    return pages

def chunk_pages(pages, chunk_size=400, overlap=50):
    """Split page text into overlapping chunks of ~chunk_size tokens."""
    chunks = []
    for page_data in pages:
        tokens = enc.encode(page_data["text"])
        start = 0
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = enc.decode(chunk_tokens)
            chunks.append({
                "source_page": page_data["page"],
                "chunk_text": chunk_text,
            })
            start += chunk_size - overlap
    return chunks
```

**Key parameters:**
- **400 tokens per chunk:** Balances specificity (small enough to be relevant) with context (large enough to contain a complete thought)
- **50-token overlap:** Ensures a concept split across a chunk boundary still appears in at least one chunk
- These are starting values — your eval suite (Phase 5) will tell you if they need tuning

#### 2.3 — Write `data/embed_and_load.py`

```python
from openai import OpenAI
from supabase import create_client
import os, json, time

client = OpenAI()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def embed_text(text):
    """Get embedding vector for a text chunk."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding

def load_chunks(chunks, expansion):
    """Embed and load chunks into Supabase."""
    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk["chunk_text"])
        supabase.table("rulebook_chunks").insert({
            "expansion": expansion,
            "source_page": chunk["source_page"],
            "chunk_text": chunk["chunk_text"],
            "embedding": embedding,
        }).execute()
        print(f"  Loaded chunk {i+1}/{len(chunks)}")
        time.sleep(0.1)  # gentle rate limiting
```

Run it and note the total cost (should be well under $0.50).

#### 2.4 — Validate with a similarity query

Run this in the SQL editor to test semantic search:

```sql
-- First, generate an embedding for your test query via Python,
-- then paste the vector here (or do this entirely in Python):

SELECT chunk_text, source_page, expansion,
       1 - (embedding <=> '[your_query_embedding]') AS similarity
FROM rulebook_chunks
ORDER BY embedding <=> '[your_query_embedding]'
LIMIT 3;
```

Or write a quick Python test:

```python
query = "When can I play Reaction cards?"
query_embedding = embed_text(query)

result = supabase.rpc("match_chunks", {
    "query_embedding": query_embedding,
    "match_count": 3,
}).execute()

for chunk in result.data:
    print(f"Page {chunk['source_page']}: {chunk['chunk_text'][:100]}...")
```

(You'll need to create the `match_chunks` Supabase function — see the Supabase pgvector docs for the boilerplate.)

### ✅ Phase 2 Checkpoint

- [ ] Both rulebook PDFs extracted and chunked successfully
- [ ] All chunks embedded and loaded into `rulebook_chunks`
- [ ] Cosine similarity query returns semantically relevant results for 2–3 test questions
- [ ] Total embedding cost logged (should be < $0.50)
- [ ] `chunk_rulebooks.py` and `embed_and_load.py` committed to repo

---

## Phase 3: Retrieval Layer & Query Router

**⏱ Time:** 3–4 hours  
**🎯 Skill focus:** Query routing, retrieval orchestration, LLM prompt engineering  
**📦 Deliverable:** A working Python module that takes a question and returns a sourced answer

### Why This Phase Matters

> The router + orchestrator pattern is the AI equivalent of a database query planner. Just like a SQL optimizer decides whether to use an index scan or a sequential scan, your router decides whether to query the card database, the vector store, or both. **The key insight: you don't always need an LLM to solve classification problems.** Rules-based routing is faster, cheaper, and more predictable.

### The Four Query Types

| # | Type | Retrieval Path | Example |
|---|------|---------------|---------|
| 1 | Card Lookup | SQL → cards table | "What does Chapel do?" |
| 2 | Filtered Search | SQL with WHERE clauses | "Show me all Action cards costing 4 or less" |
| 3 | Rules Question | Vector search → rulebook_chunks | "When can I play Reaction cards?" |
| 4 | Strategy / Combo | Both paths combined | "What combos well with Throne Room?" |

### Steps

#### 3.1 — Write `retrieval/router.py`

Build a rules-based classifier. Start simple:

```python
def classify_query(query: str) -> str:
    """Classify a query into one of four types."""
    query_lower = query.lower()

    # Type 1: Direct card lookup (mentions a specific card name)
    card_lookup_signals = ["what does", "what is", "tell me about", "describe"]
    if any(signal in query_lower for signal in card_lookup_signals):
        # Check if a known card name appears in the query
        # (load card names from DB or a cached list)
        return "card_lookup"

    # Type 2: Filtered search (comparative or list-based)
    filter_signals = ["show me", "list all", "which cards", "cards that",
                      "costing", "cost less", "cost more", "with +"]
    if any(signal in query_lower for signal in filter_signals):
        return "filtered_search"

    # Type 3: Rules question
    rules_signals = ["rule", "when can", "how does", "allowed to",
                     "can i", "what happens if", "phase", "turn order"]
    if any(signal in query_lower for signal in rules_signals):
        return "rules_question"

    # Type 4: Strategy / combo (default for complex questions)
    return "strategy_combo"
```

**Architecture decision:** Rules-based over LLM-based. Adds ~0ms latency and $0 cost per query. You can always upgrade later if evals show misclassification issues.

#### 3.2 — Write `retrieval/card_lookup.py`

```python
from supabase import create_client
import os

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def lookup_card(card_name: str) -> dict:
    """Look up a specific card by name."""
    result = supabase.table("cards") \
        .select("*") \
        .ilike("name", f"%{card_name}%") \
        .execute()
    return result.data

def filtered_search(filters: dict) -> list:
    """Search cards with filters (cost, type, expansion, attributes)."""
    query = supabase.table("cards").select("*")

    if "max_cost" in filters:
        query = query.lte("cost", filters["max_cost"])
    if "type" in filters:
        query = query.ilike("type", f"%{filters['type']}%")
    if "expansion" in filters:
        query = query.eq("expansion", filters["expansion"])
    if "min_plus_actions" in filters:
        query = query.gte("plus_actions", filters["min_plus_actions"])

    return query.execute().data
```

#### 3.3 — Write `retrieval/rules_search.py`

```python
from openai import OpenAI
from supabase import create_client
import os

client = OpenAI()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def search_rules(query: str, top_k: int = 3, expansion: str = None) -> list:
    """Semantic search over rulebook chunks."""
    # Embed the query
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
    )
    query_embedding = response.data[0].embedding

    # Call the Supabase match function
    params = {
        "query_embedding": query_embedding,
        "match_count": top_k,
    }
    if expansion:
        params["filter_expansion"] = expansion

    result = supabase.rpc("match_chunks", params).execute()
    return result.data
```

#### 3.4 — Write `retrieval/orchestrator.py`

```python
from retrieval.router import classify_query
from retrieval.card_lookup import lookup_card, filtered_search
from retrieval.rules_search import search_rules
from retrieval.synthesizer import synthesize_answer

def answer_question(query: str, expansion: str = None) -> dict:
    """Full pipeline: route → retrieve → synthesize."""
    query_type = classify_query(query)
    context = {"query_type": query_type, "sources": []}

    if query_type == "card_lookup":
        cards = lookup_card(query)  # extract card name from query
        context["sources"] = [{"type": "card_db", "data": c} for c in cards]

    elif query_type == "filtered_search":
        filters = parse_filters(query)  # extract filters from query
        cards = filtered_search(filters)
        context["sources"] = [{"type": "card_db", "data": c} for c in cards]

    elif query_type == "rules_question":
        chunks = search_rules(query, expansion=expansion)
        context["sources"] = [{"type": "rulebook", "data": c} for c in chunks]

    elif query_type == "strategy_combo":
        # Both paths
        cards = lookup_card(query)
        chunks = search_rules(query, expansion=expansion)
        context["sources"] = (
            [{"type": "card_db", "data": c} for c in cards] +
            [{"type": "rulebook", "data": c} for c in chunks]
        )

    # Synthesize final answer
    answer = synthesize_answer(query, context)
    return {"answer": answer, "sources": context["sources"], "query_type": query_type}
```

#### 3.5 — Write `retrieval/synthesizer.py`

```python
from openai import OpenAI
import os

client = OpenAI()

SYSTEM_PROMPT = """You are DominionSage, an expert on the Dominion card game.
Answer the user's question using ONLY the provided context.
Always cite your sources: mention card names and rulebook page numbers.
If the context doesn't contain enough information, say so honestly.
Keep answers concise but thorough."""

def synthesize_answer(query: str, context: dict) -> str:
    """Generate a final answer using retrieved context."""
    # Format context for the LLM
    context_text = format_context(context["sources"])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content

def format_context(sources: list) -> str:
    """Format retrieved sources into a context string for the LLM."""
    parts = []
    for s in sources:
        if s["type"] == "card_db":
            card = s["data"]
            parts.append(f"[Card: {card['name']}] Cost: {card['cost']}, "
                        f"Type: {card['type']}, Text: {card['text']}")
        elif s["type"] == "rulebook":
            chunk = s["data"]
            parts.append(f"[Rulebook p.{chunk['source_page']}, "
                        f"{chunk['expansion']}] {chunk['chunk_text']}")
    return "\n\n".join(parts)
```

**Architecture decision:** GPT-4o-mini over GPT-4o for synthesis. The retrieval quality matters more than the generation model. Save your budget for embeddings and evals.

#### 3.6 — Test from the command line

```python
# test_pipeline.py
from retrieval.orchestrator import answer_question

test_questions = [
    "What does Chapel do?",                          # Type 1
    "Show me all Action cards costing 4 or less",    # Type 2
    "When can I play Reaction cards?",               # Type 3
    "What combos well with Throne Room?",            # Type 4
]

for q in test_questions:
    print(f"\nQ: {q}")
    result = answer_question(q)
    print(f"Type: {result['query_type']}")
    print(f"A: {result['answer'][:200]}...")
    print(f"Sources: {len(result['sources'])}")
```

### ✅ Phase 3 Checkpoint

- [ ] Router correctly classifies all 4 example questions
- [ ] Card lookup returns correct card data
- [ ] Rules search returns relevant rulebook chunks
- [ ] Synthesizer produces coherent answers with source citations
- [ ] All 5 retrieval module files committed to repo

---

## Phase 4: Streamlit UI & Source Attribution

**⏱ Time:** 2–3 hours  
**🎯 Skill focus:** UI development, source attribution, user experience for AI systems  
**📦 Deliverable:** A working Streamlit app with chat interface and source panel

### Why This Phase Matters

> The source attribution panel is the single most important UI element for portfolio purposes. In production AI systems, users (and regulators) need to know WHERE an answer came from. This is the AI equivalent of **data lineage** in a data pipeline. A dashboard that shows a metric without explaining where the number came from is unreliable. An AI answer without showing its sources is the same thing. Hiring managers notice this.

### Steps

#### 4.1 — Create `app/main.py`

```python
import streamlit as st
from retrieval.orchestrator import answer_question

st.set_page_config(page_title="DominionSage", page_icon="🃏", layout="wide")
st.title("🃏 DominionSage")
st.caption("Ask anything about Dominion cards, rules, and strategy.")

# Sidebar: expansion filter
expansion = st.sidebar.selectbox(
    "Filter by expansion",
    ["All", "Base", "Seaside"],
    index=0,
)
exp_filter = None if expansion == "All" else expansion

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
    if msg["role"] == "assistant" and "sources" in msg:
        with st.expander("📋 Sources"):
            for source in msg["sources"]:
                if source["type"] == "card_db":
                    card = source["data"]
                    st.markdown(f"**🃏 Card DB:** {card['name']} "
                              f"({card['expansion']}, Cost {card['cost']})")
                elif source["type"] == "rulebook":
                    chunk = source["data"]
                    st.markdown(f"**📄 Rulebook:** {chunk['expansion']} "
                              f"p.{chunk['source_page']}")
                    st.text(chunk["chunk_text"][:200] + "...")

# Chat input
if prompt := st.chat_input("Ask about Dominion..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            result = answer_question(prompt, expansion=exp_filter)
        st.markdown(result["answer"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
    st.rerun()
```

#### 4.2 — Add error handling

Wrap the `answer_question` call in a try/except and display user-friendly error messages:

```python
try:
    result = answer_question(prompt, expansion=exp_filter)
except Exception as e:
    st.error(f"Something went wrong: {str(e)}")
```

#### 4.3 — Run it locally

```bash
streamlit run app/main.py
```

#### 4.4 — Test the full flow

Walk through these scenarios in the browser:
1. Ask "What does Chapel do?" → verify card data appears in sources
2. Ask "When can I play Reaction cards?" → verify rulebook chunks in sources
3. Ask "What combos well with Throne Room?" → verify both card + rulebook sources
4. Switch the expansion filter and verify it scopes results

### ✅ Phase 4 Checkpoint

- [ ] Streamlit app runs locally without errors
- [ ] Chat interface sends questions and displays answers
- [ ] Source attribution panel shows card DB entries and/or rulebook chunks
- [ ] Expansion filter correctly scopes results
- [ ] `app/main.py` committed to repo

---

## Phase 5: Evaluation Suite

**⏱ Time:** 2–3 hours  
**🎯 Skill focus:** AI evaluation methodology, retrieval precision, test design  
**📦 Deliverable:** 20 labeled eval questions with automated scoring and a results report

### Why This Phase Matters

> In traditional software engineering, you write unit tests to prove your code works. In AI engineering, you write **evals** to prove your system works. The difference is that AI systems are non-deterministic — the same input can produce different outputs. Evals handle this by testing **properties** ("does the answer mention the key fact?") rather than exact equality. This is the single biggest skill gap between data engineers and AI engineers. If you can measure your RAG system's quality, you are ahead of 90% of portfolio projects.

### Steps

#### 5.1 — Create `evals/questions.json`

Write 5 questions per query type (20 total). For each question, specify:

```json
[
  {
    "question": "What does Chapel do?",
    "expected_type": "card_lookup",
    "expected_source": "Chapel",
    "key_facts": ["trash", "up to 4 cards"],
    "notes": "Should find Chapel in card DB and mention trashing"
  },
  {
    "question": "When can I play Reaction cards?",
    "expected_type": "rules_question",
    "expected_source_keywords": ["reaction", "another player"],
    "key_facts": ["when another player", "action phase"],
    "notes": "Should retrieve the Reaction timing rules from Base rulebook"
  }
]
```

**Tips for writing good eval questions:**
- Include easy cases (direct card name mention) and hard cases (paraphrased questions)
- Include at least one question per type that's ambiguous — these test router edge cases
- Include one question where the answer spans multiple sources (e.g., a card + a rule)

#### 5.2 — Write `evals/run_evals.py`

```python
import json, csv, time
from retrieval.orchestrator import answer_question

with open("evals/questions.json") as f:
    questions = json.load(f)

results = []
for q in questions:
    print(f"Evaluating: {q['question']}")
    result = answer_question(q["question"])

    # Check routing accuracy
    routing_correct = result["query_type"] == q["expected_type"]

    # Check retrieval relevance (does expected source appear?)
    source_texts = " ".join([
        str(s["data"]) for s in result["sources"]
    ]).lower()
    retrieval_hit = any(
        kw.lower() in source_texts
        for kw in q.get("expected_source_keywords", [q.get("expected_source", "")])
    )

    # Check answer quality (do key facts appear?)
    answer_lower = result["answer"].lower()
    facts_found = [f for f in q["key_facts"] if f.lower() in answer_lower]
    answer_quality = len(facts_found) / len(q["key_facts"]) if q["key_facts"] else 1.0

    results.append({
        "question": q["question"],
        "expected_type": q["expected_type"],
        "actual_type": result["query_type"],
        "routing_correct": routing_correct,
        "retrieval_hit": retrieval_hit,
        "answer_quality": round(answer_quality, 2),
        "facts_found": "; ".join(facts_found),
    })
    time.sleep(1)  # rate limiting

# Write results
with open("evals/results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

# Print summary
routing_acc = sum(r["routing_correct"] for r in results) / len(results)
retrieval_prec = sum(r["retrieval_hit"] for r in results) / len(results)
avg_quality = sum(r["answer_quality"] for r in results) / len(results)

print(f"\n{'='*50}")
print(f"EVAL RESULTS ({len(results)} questions)")
print(f"{'='*50}")
print(f"Routing accuracy:     {routing_acc:.0%}")
print(f"Retrieval precision:  {retrieval_prec:.0%}")
print(f"Answer quality:       {avg_quality:.0%}")
```

#### 5.3 — Run evals and iterate

```bash
python evals/run_evals.py
```

**Target benchmarks:**
- Routing accuracy: ≥ 85%
- Retrieval precision (at k=3): ≥ 80%
- Answer quality: ≥ 75%

**If routing accuracy is low:** Add more keyword rules to the router. Log the misclassified questions and look for patterns.

**If retrieval precision is low:** Try smaller chunk sizes (300 or 200 tokens). Add expansion metadata as a pre-filter. Check if pdfplumber is extracting text correctly from multi-column pages.

**If answer quality is low:** Check whether the correct context IS being retrieved (retrieval problem) or whether the LLM is ignoring the context (prompt problem). These require different fixes.

### ✅ Phase 5 Checkpoint

- [ ] 20 eval questions written across all 4 query types
- [ ] Eval runner produces results.csv with per-question scores
- [ ] Aggregate metrics printed (routing accuracy, retrieval precision, answer quality)
- [ ] At least one iteration of improvement based on eval results
- [ ] All eval files committed to repo

---

## Phase 6: Deployment & README

**⏱ Time:** 1–2 hours  
**🎯 Skill focus:** Technical communication, project packaging, deployment  
**📦 Deliverable:** A polished GitHub repo with README, architecture diagram, and eval results

### Why This Phase Matters

> Hiring managers spend 30 seconds on a repo before deciding whether to look deeper. Your README is the first 30 seconds. The "What I'd do next" section is especially powerful — it shows you see beyond the MVP and understand where production RAG systems are heading.

### Steps

#### 6.1 — Write `README.md`

Use this exact structure:

**Section 1 — What it does** (one paragraph, plain English, no jargon)

> DominionSage is a Dominion card game assistant that answers questions about cards, rules, combos, and strategy using a hybrid retrieval approach — combining a structured PostgreSQL card database with semantic search over rulebook text via vector embeddings.

**Section 2 — Architecture diagram** (ASCII or Mermaid)

```
User Question
      │
      ▼
┌──────────┐
│  Router   │ ← classifies query type
└────┬─────┘
     │
     ├── Type 1/2 ──▶ Card Database (SQL)
     │                      │
     ├── Type 3 ────▶ Vector Store (pgvector)
     │                      │
     └── Type 4 ────▶ Both paths
                            │
                            ▼
                   ┌──────────────┐
                   │ LLM Synthesis │
                   │ (GPT-4o-mini) │
                   └──────┬───────┘
                          │
                          ▼
                  Answer + Sources
```

**Section 3 — Why hybrid retrieval** (3 sentences)

> Explain that card lookups need precision (exact name, cost filters) while rules questions need semantic understanding. Vector-only would lose structured relationships. SQL-only can't handle open-ended questions.

**Section 4 — Eval results**

> "Achieved X% routing accuracy and Y% retrieval precision at k=3 on 20 hand-labeled questions across 4 query types."

**Section 5 — What I'd do next**

List 3–4 future improvements:
- BM25 + vector hybrid search (reciprocal rank fusion)
- Fine-tuned embeddings on card text
- Expansion to all 14+ Dominion expansions
- Cross-game generalization (MTG, Ark Nova)

#### 6.2 — Create `requirements.txt`

```
supabase>=2.0.0
openai>=1.0.0
pdfplumber>=0.10.0
streamlit>=1.30.0
tiktoken>=0.5.0
python-dotenv>=1.0.0
```

#### 6.3 — Create `.env.example`

```
SUPABASE_URL=your-supabase-url
SUPABASE_KEY=your-supabase-anon-key
OPENAI_API_KEY=your-openai-key
```

#### 6.4 — Deploy to Streamlit Community Cloud

1. Push everything to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo
4. Set environment variables in the Streamlit dashboard
5. Deploy

#### 6.5 — Final review

- [ ] Clone the repo to a fresh directory and follow the README — does it work?
- [ ] Are there any hardcoded paths or credentials?
- [ ] Is the `.env` file in `.gitignore`?

### ✅ Phase 6 Checkpoint

- [ ] README follows the 5-section structure
- [ ] Architecture diagram included
- [ ] Eval results cited with specific numbers
- [ ] App deployed and accessible via URL
- [ ] A stranger can clone, follow the README, and run the project

---

## Quick Reference: Architecture Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Database | Supabase (PostgreSQL + pgvector) | Free hosted, vector search built in, Python client |
| Embedding model | text-embedding-3-small | Best cost/quality for small corpus; upgrade path to large |
| LLM | GPT-4o-mini | Fast + cheap; retrieval quality matters more |
| Router | Rules-based | 0ms latency, $0 cost, sufficient for 4 query types |
| Chunk size | ~400 tokens, 50 overlap | Standard starting point; tunable via evals |
| PDF extraction | pdfplumber | Handles multi-column layouts |
| UI | Streamlit | Ship in hours, not days |
| Deployment | Streamlit Community Cloud | Free, direct GitHub integration |

---

## Quick Reference: Estimated Costs

| Item | Cost |
|------|------|
| Supabase | Free tier |
| OpenAI embeddings | < $0.50 |
| OpenAI GPT-4o-mini (all dev + evals) | < $2.00 |
| Streamlit deployment | Free |
| **Total** | **< $3.00** |

---

## What To Say in Interviews

When someone asks about this project, hit these points:

1. **"I used hybrid retrieval because..."** — not all data belongs in a vector store. Structured queries are faster and more precise when you have known attributes.

2. **"I measured it with evals because..."** — any RAG demo can look good with cherry-picked questions. Evals prove the system works across query types.

3. **"The source attribution panel shows..."** — production AI systems need to be auditable. Users need to know where answers come from.

4. **"I'd improve it by adding..."** — BM25 + vector hybrid search, conversation memory, and cross-game expansion. This shows you understand the trajectory of the field.
