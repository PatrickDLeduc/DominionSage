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
from retrieval.agent import answer_question_agent
from retrieval.card_lookup import (
    get_all_kingdom_card_names,
    get_kingdom_cards_by_names,
    get_all_expansion_names,
    get_random_kingdom,
)
from retrieval.kingdom_advisor import analyze_kingdom
from data.kingdom_presets import KINGDOM_PRESETS
from simulation.runner import run_simulation, analyze_simulation
from simulation.bots import AVAILABLE_BOTS
from app.rate_limit import check_rate_limit


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

    limit_cards = st.checkbox(
        "Limit card results to 20",
        value=True,
        help="Uncheck to return all matching cards for broad searches. Warning: Returning 100+ cards may slow down generation.",
    )

    st.divider()

    use_agent = st.checkbox(
        "Agent mode",
        value=True,
        help="Use an LLM agent that dynamically selects tools instead of the rule-based router. Handles complex multi-part questions better.",
    )

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

        **Agent mode** (default): An LLM agent dynamically decides which
        tools to call — card lookup, filtered search, rules search, or
        strategy search — and can use multiple tools per query.

        **Pipeline mode**: A rule-based router classifies your question
        into one of 4 types and follows a fixed retrieval path.

        Both modes use **Card DB** (PostgreSQL), **Vector Store** (pgvector),
        and **GPT-4o-mini** for synthesis. Every answer shows its sources.
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

chat_tab, kingdom_tab, sim_tab = st.tabs(["💬 Chat", "🏰 Kingdom Advisor", "🎮 Simulator"])


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

    # Display chat history in a scrollable container
    chat_container = st.container(height=600)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

            # Show citations and sources after assistant messages
            if msg["role"] == "assistant":
                # Meta notes
                for note in msg.get("meta_notes", []):
                    st.info(f"**Note:** {note}")

                # Structured citations
                msg_citations = msg.get("citations", [])
                msg_source_map = msg.get("source_map", {})
                if msg_citations and msg_source_map:
                    with st.expander(f"📎 {len(msg_citations)} citation(s)"):
                        for cite in msg_citations:
                            label = cite.get("source_label", "")
                            display = msg_source_map.get(label, {}).get("display", label)
                            st.markdown(f"- **{display}**: {cite.get('claim', '')}")

                # Source panel
                if "sources" in msg:
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
        
        with chat_container:
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
        with chat_container:
            with st.chat_message("assistant"):
                if use_agent:
                    status = st.status("Thinking...", expanded=False)
                    def _on_tool_call(tool_name, args):
                        tool_labels = {
                            "lookup_card": f"Looking up {args.get('card_name', 'card')}...",
                            "search_cards": "Searching card database...",
                            "search_rules": "Searching rulebooks...",
                            "search_strategy": "Searching strategy guides...",
                        }
                        status.update(label=tool_labels.get(tool_name, f"Using {tool_name}..."))
                    try:
                        if not check_rate_limit(limit=4, window=60):
                            raise Exception("Rate limit exceeded (4 requests per minute). Please wait a moment before trying again.")

                        result = answer_question_agent(
                            prompt,
                            expansion=exp_filter,
                            conversation_history=conversation_history,
                            kingdom_context=kingdom_ctx,
                            limit_cards=limit_cards,
                            on_tool_call=_on_tool_call,
                        )
                        answer = result["answer"]
                        sources = result["sources"]
                        query_type = result["query_type"]
                    except Exception as e:
                        print(f"Error during agent query: {e}")
                        answer = "Sorry, our servers are currently busy or encountered an error. Please try again in a moment."
                        sources = []
                        query_type = "error"
                        result = {}
                    finally:
                        status.update(label="Done", state="complete")
                else:
                    with st.spinner("Searching cards and rulebooks..."):
                        try:
                            if not check_rate_limit(limit=4, window=60):
                                raise Exception("Rate limit exceeded (4 requests per minute). Please wait a moment before trying again.")

                            result = answer_question(
                                prompt,
                                expansion=exp_filter,
                                conversation_history=conversation_history,
                                kingdom_context=kingdom_ctx,
                                limit_cards=limit_cards,
                            )
                            answer = result["answer"]
                            sources = result["sources"]
                            query_type = result["query_type"]
                        except Exception as e:
                            print(f"Error during chat query: {e}")
                            answer = "Sorry, our servers are currently busy or encountered an error. Please try again in a moment."
                            sources = []
                            query_type = "error"
                            result = {}

                # Show rewrite indicator if the query was rewritten
                if "rewritten_query" in result:
                    st.caption(f"🔄 Interpreted as: *{result['rewritten_query']}*")

                st.markdown(answer)

                # Show meta notes as info callouts
                for note in result.get("meta_notes", []):
                    st.info(f"**Note:** {note}")

                # Show structured citations
                citations = result.get("citations", [])
                source_map = result.get("source_map", {})
                if citations and source_map:
                    with st.expander(f"📎 {len(citations)} citation(s)"):
                        for cite in citations:
                            label = cite.get("source_label", "")
                            display = source_map.get(label, {}).get("display", label)
                            st.markdown(f"- **{display}**: {cite.get('claim', '')}")

                # Show sources
                if sources:
                    render_sources(sources)

        # Save to session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "query_type": query_type,
            "citations": result.get("citations", []),
            "source_map": result.get("source_map", {}),
            "meta_notes": result.get("meta_notes", []),
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

    # Load card names for the dropdown (cached indefinitely to avoid repeated DB calls)
    @st.cache_data
    def _load_card_names():
        return get_all_kingdom_card_names()

    @st.cache_data
    def _load_expansion_names():
        return get_all_expansion_names()

    try:
        all_card_names = _load_card_names()
    except Exception as e:
        print(f"Could not load card names: {e}")
        st.error("Could not reach the database to load card names. Please try again later.")
        all_card_names = []

    try:
        all_expansions = _load_expansion_names()
    except Exception as e:
        print(f"Could not load expansion names: {e}")
        all_expansions = []

    # ── Quick-pick: Presets & Random ─────────────────────────────
    st.subheader("⚡ Quick Pick")
    qp_col1, qp_col2 = st.columns(2)

    with qp_col1:
        # Build grouped options for the preset selectbox
        preset_options = {f"{p['name']}  ({' + '.join(p['expansions'])})": p for p in KINGDOM_PRESETS}
        preset_choice = st.selectbox(
            "📖 Official Preset",
            options=[""] + list(preset_options.keys()),
            index=0,
            help="Official recommended sets from the Dominion rulebooks.",
            key="preset_selectbox",
        )
        if st.button("Load Preset", disabled=not preset_choice, key="load_preset_btn"):
            preset = preset_options[preset_choice]
            st.session_state["kingdom_multiselect"] = preset["cards"]
            st.rerun()

    with qp_col2:
        owned_expansions = st.multiselect(
            "🎲 Random from Expansions",
            options=all_expansions,
            placeholder="Select your expansions...",
            help="Pick the expansions you own, then generate a random kingdom.",
            key="owned_expansions",
        )
        if st.button("Generate Random", disabled=len(owned_expansions) == 0, key="random_kingdom_btn"):
            random_cards = get_random_kingdom(owned_expansions)
            if len(random_cards) < 10:
                st.warning(f"Only {len(random_cards)} eligible cards found — need 10.")
            else:
                st.session_state["kingdom_multiselect"] = random_cards
                st.rerun()

    st.divider()

    # ── Card selection ───────────────────────────────────────────
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
                if not check_rate_limit(limit=4, window=60):
                    raise Exception("Rate limit exceeded (4 requests per minute). Please wait a moment before trying again.")

                # Fetch full card data
                cards = get_kingdom_cards_by_names(selected_cards)

                # Run GPT analysis (returns KingdomAdvisorResponse)
                advice = analyze_kingdom(cards)

                # Store as dict for Streamlit serialization
                st.session_state["kingdom_advice"] = advice.model_dump()
                st.session_state["kingdom_cards_data"] = cards
            except Exception as e:
                print(f"Error during kingdom analysis: {e}")
                st.error("Analysis failed due to a server error. Please try again in a moment.")

    # Display results (persisted across reruns via session state)
    if "kingdom_advice" in st.session_state and st.session_state.get("kingdom_multiselect"):
        st.divider()

        # Render structured strategy advice
        advice = st.session_state["kingdom_advice"]

        st.subheader("🎯 Opening Strategy")
        opening = advice.get("opening_strategy", {})
        st.markdown(f"**3/4 Split:** {opening.get('three_four_split', 'N/A')}")
        st.markdown(f"**5/2 Split:** {opening.get('five_two_split', 'N/A')}")

        st.subheader("🔗 Key Combos")
        for combo in advice.get("key_combos", []):
            cards_str = " + ".join(f"**{c}**" for c in combo.get("cards", []))
            st.markdown(f"{cards_str}: {combo.get('explanation', '')}")

        st.subheader("📊 Archetype Assessment")
        st.markdown(advice.get("archetype_assessment", "N/A"))

        st.subheader("⚠️ Cards to Avoid")
        for card_note in advice.get("cards_to_avoid", []):
            st.markdown(f"- {card_note}")

        st.subheader("🛡️ Attack & Defense")
        st.markdown(advice.get("attack_and_defense", "N/A"))

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


# ─────────────────────────────────────────────────────────────────
# Tab 3: Simulator
# ─────────────────────────────────────────────────────────────────

with sim_tab:
    st.header("🎮 Bot Simulator")
    st.markdown(
        "Run bot-vs-bot Dominion simulations to discover strategies through data. "
        "Two bots play hundreds of games, and AI analyzes the results for insights."
    )

    # Get kingdom from the advisor tab, or use First Game as default
    sim_kingdom = st.session_state.get("kingdom_multiselect", [])
    if not sim_kingdom or len(sim_kingdom) != 10:
        sim_kingdom = ["Cellar", "Market", "Merchant", "Militia", "Mine",
                       "Moat", "Remodel", "Smithy", "Village", "Workshop"]

    st.info(f"**Kingdom**: {', '.join(sim_kingdom)}")
    st.caption("Select cards in the Kingdom Advisor tab to simulate a different kingdom.")

    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        n_games = st.slider("Number of games", 50, 1000, 200, step=50,
                            key="sim_n_games")
        num_players = st.slider("Number of players", 2, 4, 2, key="sim_num_players")
    with sim_col2:
        bot_names = list(AVAILABLE_BOTS.keys())
        selected_bots = []
        for i in range(num_players):
            default_index = i % len(bot_names)
            selected_bots.append(st.selectbox(f"Bot {i+1} Strategy", bot_names, index=default_index))

    if st.button("🚀 Run Simulation", key="run_sim_btn", type="primary"):
        with st.spinner(f"Simulating {n_games} games..."):
            bot_classes = [AVAILABLE_BOTS[name] for name in selected_bots]
            stats = run_simulation(sim_kingdom, n_games=n_games, 
                                   bot_classes=bot_classes)
            st.session_state["sim_stats"] = stats
            # Clear previous analysis so it doesn't show stale results
            st.session_state.pop("sim_analysis", None)

    # Display results
    if "sim_stats" in st.session_state:
        stats = st.session_state["sim_stats"]
        st.divider()

        # Win rates
        st.subheader("📊 Results")
        cols = st.columns(len(stats["bot_names"]) + 1)
        for i, name in enumerate(stats["bot_names"]):
            with cols[i]:
                win_pct = stats["win_rates"][name]
                st.metric(f"{name}", f"{stats['wins'][name]} wins",
                          delta=f"{win_pct}%")
        with cols[-1]:
            st.metric("Ties", stats["ties"])

        # VP and turns
        vp_cols = st.columns(len(stats["bot_names"]) + 1)
        for i, name in enumerate(stats["bot_names"]):
            with vp_cols[i]:
                st.metric(f"Avg VP ({name})", stats["avg_vp"][name])
        with vp_cols[-1]:
            st.metric("Avg Turns", stats["avg_turns"])

        # Buy frequency
        st.subheader("🛒 Average Cards Purchased per Game")
        buy_cols = st.columns(len(stats["bot_names"]))
        for i, name in enumerate(stats["bot_names"]):
            with buy_cols[i]:
                st.markdown(f"**{name}**")
                buys = stats["buy_frequency"][name]
                for card, avg in buys.items():
                    if avg >= 0.1:
                        bar_len = int(avg * 3)
                        st.markdown(f"`{card:20s}` {'█' * bar_len} {avg}")

        # GPT Analysis
        st.divider()
        st.subheader("🧠 AI Strategy Analysis")
        if "sim_analysis" not in st.session_state:
            if st.button("Analyze with AI", key="analyze_sim_btn"):
                with st.spinner("AI is analyzing the simulation data..."):
                    try:
                        if not check_rate_limit(limit=4, window=60):
                            raise Exception("Rate limit exceeded (4 requests per minute). Please wait a moment before trying again.")

                        analysis = analyze_simulation(stats)
                        # Store as dict for Streamlit serialization
                        st.session_state["sim_analysis"] = analysis.model_dump()
                        st.rerun()
                    except Exception as e:
                        print(f"Error during simulation analysis: {e}")
                        st.error("Analysis failed due to a server error. Please try again in a moment.")
        else:
            # Render structured analysis sections
            analysis = st.session_state["sim_analysis"]
            sections = [
                ("🏆 Why the Winner Wins", analysis.get("why_winner_wins", "")),
                ("🔗 Key Card Interactions", analysis.get("key_card_interactions", "")),
                ("🎯 Optimal Opening", analysis.get("optimal_opening", "")),
                ("💡 Strategic Recommendations", analysis.get("strategic_recommendations", "")),
                ("🔍 Surprising Findings", analysis.get("surprising_findings", "")),
            ]
            for heading, content in sections:
                if content:
                    st.markdown(f"**{heading}**")
                    st.markdown(content)
                    st.divider()

            if st.button("Re-analyze", key="reanalyze_btn"):
                del st.session_state["sim_analysis"]
                st.rerun()