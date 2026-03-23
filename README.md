# 🃏 DominionSage

**An AI-powered Dominion card game assistant using hybrid retrieval — combining a structured PostgreSQL database of 500+ cards from all 15 expansions with semantic search over rulebook text via vector embeddings.**

Ask about cards, rules, combos, and strategy. Every answer shows its sources so you can verify the information.

🔗 **[Try it live →](https://dominionsage.streamlit.app/)**

---

## Features

1. **💬 AI Chat Assistant**: Ask about card rules, combos, or strategies. The hybrid retrieval engine pulls exact card text from the PostgreSQL database and semantic rulings from the vectorized rulebooks.
2. **🏰 Kingdom Advisor**: Build a custom kingdom, load an official rulebook preset, or generate a random set from your owned expansions. The selected kingdom is shared with the Chat tab so the AI knows exactly what cards are in play.
3. **🎮 Bot Simulator**: Run bot-vs-bot games (Big Money vs Engine) on any subset of the 26 Base Set cards. Play hundreds of games in milliseconds, view buy frequencies and win rates, and have GPT-4o-mini analyze the results for strategic insights.

---

## Architecture

```
User Question
      │
      ▼
┌──────────┐
│  Router   │ ← rules-based classifier (0ms, $0/query)
└────┬─────┘
     │
     ├── Card Lookup ────────▶ Card Database (PostgreSQL)
     │   "What does Chapel do?"        │
     │                                 │
     ├── Filtered Search ───▶ Card Database (SQL WHERE)
     │   "Show me all 4-cost Actions"  │
     │                                 │
     ├── Rules Question ────▶ Vector Store (pgvector)
     │   "When can I play Reactions?"   │
     │                                 │
     └── Strategy / Combo ──▶ Both paths
         "What combos with Throne Room?"│
                                       ▼
                              ┌──────────────┐
                              │ LLM Synthesis │
                              │ (GPT-4o-mini) │
                              └──────┬───────┘
                                     │
                                     ▼
                             Answer + Sources
```

## Why Hybrid Retrieval?

Card lookups need precision — exact name matches, cost filters, attribute queries. Vector-only retrieval would lose these structured relationships and force every query through an embedding + similarity search, even when a simple `WHERE cost <= 4` is faster and more accurate. Conversely, SQL alone can't handle open-ended rules questions like "When can I play Reaction cards?" where the answer requires semantic understanding of rulebook text. The hybrid approach uses the right tool for each query type.

## Eval Results

Evaluated on 20 hand-labeled questions across all 4 query types:

| Metric | Score | Target |
|--------|-------|--------|
| Routing accuracy | 90%| ≥ 85% |
| Retrieval precision (k=3) | 100% | ≥ 80% |
| Answer quality | 86% | ≥ 75% |

The eval suite tests routing correctness (did the query go to the right retrieval path?), retrieval relevance (did the correct source appear in the top results?), and answer quality (does the generated answer contain key facts?). Scoring uses normalized substring matching to handle variants like "clean-up" vs "cleanup".

**Known limitations:**
- Page citations in rulebook answers can be imprecise — the LLM cites the page number from the retrieved chunk, but that chunk may not be the most specific source for every claim in the answer.
- Key-fact scoring is substring-based and can't catch paraphrased facts. An LLM-as-judge scorer would improve this.

## What I'd Do Next

- **BM25 + vector hybrid search** — combine keyword matching with semantic search using reciprocal rank fusion, eliminating the failure mode where the correct chunk contains the right keywords but has a low vector similarity score.
- **LLM-as-judge evaluation** — use a more powerful model to score answer quality semantically, enabling evaluation at scale without hand-labeling.
- **Conversation memory** — add a short-term context window for follow-up questions like "What about with Village?" after asking about Throne Room combos.
- **Cross-game generalization** — the hybrid retrieval architecture is game-agnostic and could support Magic: The Gathering, Ark Nova, or other card games.

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Database | Supabase (PostgreSQL + pgvector) | Free hosted, vector search built in, Python client |
| Embeddings | OpenAI text-embedding-3-small | Best cost/quality for small corpus |
| LLM | GPT-4o-mini | Fast + cheap; retrieval quality matters more than generation model |
| Router | Rules-based (Python) | 0ms latency, $0 cost, sufficient for 4 query types |
| UI | Streamlit | Ship in hours, not days |
| Deployment | Streamlit Community Cloud | Free, direct GitHub integration |

**Total project cost: < $3.00** (embeddings + LLM calls during development and eval runs).

## Run It Yourself

### Prerequisites
- Python 3.11+
- A [Supabase](https://supabase.com) account (free tier)
- An [OpenAI](https://platform.openai.com) API key

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/dominionsage.git
cd dominionsage
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file from the template:

```bash
cp .env.example .env
# Edit .env with your Supabase and OpenAI credentials
```

### Database Setup

1. Create a Supabase project and enable pgvector:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

2. Create tables (see `match_chunks.sql` for the full schema).

3. Ingest card data:
   ```bash
   python data/scrape_cards.py
   python data/load_cards.py --validate
   ```

4. Chunk and embed rulebooks:
   ```bash
   python data/chunk_rulebooks.py
   python data/embed_and_load.py --validate
   ```

### Run Locally

```bash
streamlit run app/main.py
```

### Run Evals

```bash
python evals/run_evals.py
```

## Project Structure

```
dominionsage/
├── app/
│   └── main.py              # Streamlit chat UI with source attribution
├── data/
│   ├── scrape_cards.py      # Card data scraping + attribute extraction
│   ├── load_cards.py        # Card loading into Supabase
│   ├── chunk_rulebooks.py   # PDF extraction + token-based chunking
│   └── embed_and_load.py    # OpenAI embedding + pgvector loading
├── retrieval/
│   ├── router.py            # Rules-based query classifier (4 types)
│   ├── card_lookup.py       # SQL card queries
│   ├── rules_search.py      # pgvector semantic search
│   ├── orchestrator.py      # Multi-path retrieval coordinator
│   └── synthesizer.py       # GPT-4o-mini answer generation
├── evals/
│   ├── questions.json       # 20 hand-labeled eval questions
│   ├── run_evals.py         # Automated scoring (routing, retrieval, quality)
│   └── results.csv          # Eval output
├── simulation/
│   ├── engine.py            # Core game state & turn loop
│   ├── cards.py             # Card definitions & effects (Base Set)
│   ├── bots.py              # AI strategies (Big Money, Engine)
│   └── runner.py            # Orchestrator & GPT analysis
├── match_chunks.sql         # Supabase RPC functions for vector search
├── requirements.txt
├── .env.example
└── README.md
```
