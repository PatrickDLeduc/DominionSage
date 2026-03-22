"""
app/main.py — DominionSage Streamlit UI (Phase 4)

A chat interface for asking questions about Dominion cards, rules,
and strategy. Includes a source attribution panel that shows exactly
where each answer came from — the single most important UI element
for a portfolio RAG project.

Now also features a Kingdom Advisor tab — input 10 kingdom cards
and get a personalized opening strategy analysis.

Usage:
  streamlit run app/main.py
"""

import streamlit as st
import sys
import os

# Add project root to path so we can import the retrieval package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.orchestrator import answer_question
from retrieval.card_lookup import get_all_kingdom_card_names, get_kingdom_cards_by_names
from retrieval.kingdom_advisor import analyze_kingdom


# ─────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DominionSage",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🃏 DominionSage")
    st.caption("AI-powered Dominion card game assistant")

    st.divider()

    # Expansion filter
    expansion = st.selectbox(
        "Filter by expansion",
        [
            "All",
            "Base",
            "Seaside",
            "Intrigue",
            "Prosperity",
            "Hinterlands",
            "Dark Ages",
            "Adventures",
            "Empires",
            "Nocturne",
            "Renaissance",
            "Menagerie",
            "Alchemy",
            "Cornucopia",
            "Guilds",
            "Allies",
            "Plunder",
        ],
        index=0,
        help="Scope answers to a specific expansion, or search across all.",
    )
    exp_filter = None if expansion == "All" else expansion

    st.divider()

    # Example questions
    st.markdown("**Try asking:**")
    example_questions = [
        "What does Chapel do?",
        "Show me all Duration cards",
        "When can I play Reaction cards?",
        "What combos well with Throne Room?",
    ]
    for eq in example_questions:
        if st.button(eq, use_container_width=True):
            st.session_state["prefill_question"] = eq

    st.divider()

    # Architecture info (shows hiring managers you know what you built)
    with st.expander("🏗️ How it works"):
        st.markdown("""
        **DominionSage** uses hybrid retrieval:

        1. **Router** classifies your question into one of 4 types
        2. **Card DB** (PostgreSQL) handles card lookups and filtered searches
        3. **Vector Store** (pgvector) handles rules questions via semantic search
        4. **GPT-4o-mini** synthesizes the final answer from retrieved context

        Every answer shows its sources so you can verify the information.
        """)

    # Clear chat button
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ─────────────────────────────────────────────────────────────────
# Chat state
# ─────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []


# ─────────────────────────────────────────────────────────────────
# Source attribution display
# ─────────────────────────────────────────────────────────────────

def render_sources(sources: list[dict]) -> None:
    """
    Render the source attribution panel for an answer.

    This is the most important UI element for a portfolio RAG project.
    It shows hiring managers that you understand AI systems need to be
    auditable — users need to know WHERE answers come from.
    """
    # Filter out meta sources (those are already appended to the answer)
    real_sources = [s for s in sources if s["type"] != "meta"]

    if not real_sources:
        return

    # Count by type
    card_sources = [s for s in real_sources if s["type"] == "card_db"]
    rule_sources = [s for s in real_sources if s["type"] == "rulebook"]

    # Build label
    parts = []
    if card_sources:
        parts.append(f"{len(card_sources)} card(s)")
    if rule_sources:
        parts.append(f"{len(rule_sources)} rulebook chunk(s)")
    label = f"📋 Sources: {', '.join(parts)}"

    with st.expander(label):
        # Card sources
        if card_sources:
            st.markdown("**🃏 Card Database**")
            for source in card_sources:
                card = source["data"]
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(
                        f"**{card['name']}**  \n"
                        f"Cost: {card.get('cost', '?')} | "
                        f"{card.get('type', '?')}  \n"
                        f"*{card.get('expansion', '?')}*"
                    )
                with col2:
                    st.text(card.get("text", "N/A"))
                st.divider()

        # Rulebook sources
        if rule_sources:
            st.markdown("**📄 Rulebook Chunks**")
            for source in rule_sources:
                chunk = source["data"]
                similarity = chunk.get("similarity", 0)
                sim_pct = f"{similarity:.0%}" if similarity else "N/A"

                st.markdown(
                    f"**{chunk.get('expansion', '?')}** — "
                    f"Page {chunk.get('source_page', '?')} "
                    f"(relevance: {sim_pct})"
                )
                st.text(chunk.get("chunk_text", "N/A")[:300])
                if len(chunk.get("chunk_text", "")) > 300:
                    st.caption("(truncated)")
                st.divider()


# ─────────────────────────────────────────────────────────────────
# Tabbed layout: Chat + Kingdom Advisor
# ─────────────────────────────────────────────────────────────────

chat_tab, kingdom_tab = st.tabs(["💬 Chat", "🏰 Kingdom Advisor"])


# ─────────────────────────────────────────────────────────────────
# Tab 1: Chat (existing functionality)
# ─────────────────────────────────────────────────────────────────

with chat_tab:
    # Show kingdom context indicator if cards are selected
    active_kingdom = st.session_state.get("kingdom_multiselect", [])
    if active_kingdom:
        st.caption(
            f"🏰 Kingdom active: {', '.join(active_kingdom)} — "
            f"Chat answers will consider these cards."
        )

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

        # Show sources after assistant messages
        if msg["role"] == "assistant" and "sources" in msg:
            render_sources(msg["sources"])

    # Check for prefilled question from sidebar buttons
    prefill = st.session_state.pop("prefill_question", None)
    prompt = st.chat_input("Ask about Dominion cards, rules, or strategy...")

    # Use prefilled question if a sidebar button was clicked
    if prefill:
        prompt = prefill

    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Build conversation history for the rewriter
        # Take only user/assistant content pairs (skip sources, query_type, etc.)
        conversation_history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.messages[:-1]  # exclude the message we just added
            if msg["role"] in ("user", "assistant")
        ]

        # Build kingdom context if cards are selected in the Kingdom Advisor
        kingdom_ctx = None
        selected_kingdom = st.session_state.get("kingdom_multiselect", [])
        if selected_kingdom:
            kingdom_ctx = ", ".join(selected_kingdom)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching cards and rulebooks..."):
                try:
                    result = answer_question(
                        prompt,
                        expansion=exp_filter,
                        conversation_history=conversation_history,
                        kingdom_context=kingdom_ctx,
                    )
                    answer = result["answer"]
                    sources = result["sources"]
                    query_type = result["query_type"]
                except Exception as e:
                    answer = f"Sorry, something went wrong: {str(e)}"
                    sources = []
                    query_type = "error"
                    result = {}

            # Show rewrite indicator if the query was rewritten
            if "rewritten_query" in result:
                st.caption(f"🔄 Interpreted as: *{result['rewritten_query']}*")

            st.markdown(answer)

        # Show sources
        if sources:
            render_sources(sources)

        # Save to session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "query_type": query_type,
        })


# ─────────────────────────────────────────────────────────────────
# Tab 2: Kingdom Advisor
# ─────────────────────────────────────────────────────────────────

with kingdom_tab:
    st.header("🏰 Kingdom Advisor")
    st.markdown(
        "Enter the 10 kingdom cards for your game, and get a personalized "
        "strategy breakdown — opening buys, key combos, and archetype advice."
    )

    # Load card names for the dropdown (cached to avoid repeated DB calls)
    @st.cache_data(ttl=600)
    def _load_card_names():
        return get_all_kingdom_card_names()

    try:
        all_card_names = _load_card_names()
    except Exception as e:
        st.error(f"Could not load card names: {e}")
        all_card_names = []

    # Card selection
    selected_cards = st.multiselect(
        "Select 10 kingdom cards",
        options=all_card_names,
        max_selections=10,
        placeholder="Type to search cards...",
        help="Search and select exactly 10 kingdom cards for your game.",
        key="kingdom_multiselect",
    )

    # Card count indicator
    count = len(selected_cards)
    if count == 0:
        st.info("Select 10 kingdom cards to get started.")
    elif count < 10:
        st.warning(f"**{count}/10** cards selected — add {10 - count} more.")
    else:
        st.success(f"**{count}/10** cards selected — ready to analyze!")

    # Analyze button
    col_btn, col_spacer = st.columns([1, 3])
    with col_btn:
        analyze_clicked = st.button(
            "⚔️ Analyze Kingdom",
            disabled=(count != 10),
            use_container_width=True,
            type="primary",
        )

    if analyze_clicked and count == 10:
        with st.spinner("Analyzing kingdom strategy..."):
            try:
                # Fetch full card data
                cards = get_kingdom_cards_by_names(selected_cards)

                # Run GPT analysis
                advice = analyze_kingdom(cards)

                # Store results
                st.session_state["kingdom_advice"] = advice
                st.session_state["kingdom_cards_data"] = cards
            except Exception as e:
                st.error(f"Analysis failed: {e}")

    # Display results (persisted across reruns via session state)
    if "kingdom_advice" in st.session_state and st.session_state.get("kingdom_multiselect"):
        st.divider()

        # Strategy advice
        st.markdown(st.session_state["kingdom_advice"])

        # Card reference grid
        st.divider()
        st.subheader("📋 Card Reference")

        cards_data = st.session_state.get("kingdom_cards_data", [])
        if cards_data:
            # Display cards in a 2-column grid
            for i in range(0, len(cards_data), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(cards_data):
                        card = cards_data[idx]
                        with col:
                            with st.container(border=True):
                                st.markdown(
                                    f"**{card['name']}** — "
                                    f"Cost: {card.get('cost', '?')} | "
                                    f"{card.get('type', '?')}"
                                )
                                card_text = card.get("text", "")
                                if card_text:
                                    st.caption(card_text[:150] + ("..." if len(card_text) > 150 else ""))
                                else:
                                    st.caption("(No card text)")
                                # Show stats bar
                                stats = []
                                if card.get("plus_actions", 0):
                                    stats.append(f"+{card['plus_actions']} Actions")
                                if card.get("plus_cards", 0):
                                    stats.append(f"+{card['plus_cards']} Cards")
                                if card.get("plus_buys", 0):
                                    stats.append(f"+{card['plus_buys']} Buys")
                                if card.get("plus_coins", 0):
                                    stats.append(f"+{card['plus_coins']} Coins")
                                if stats:
                                    st.markdown(f"*{' · '.join(stats)}*")