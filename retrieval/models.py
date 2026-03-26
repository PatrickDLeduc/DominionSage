"""
models.py — Pydantic response schemas for structured LLM output.

These models are used with OpenAI's structured output feature
(client.beta.chat.completions.parse) to guarantee valid, typed
responses from GPT-4o-mini. No more regex parsing or hoping the
LLM follows formatting instructions.
"""

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────
# Synthesizer (chat answer generation)
# ─────────────────────────────────────────────────────────────────

class Citation(BaseModel):
    source_label: str = Field(
        description="The source label referenced, e.g. 'Source 1', 'Source 2'"
    )
    claim: str = Field(
        description="The specific claim or sentence this citation supports"
    )


class SynthesizerResponse(BaseModel):
    answer: str = Field(
        description=(
            "The complete answer in markdown. Use bold for card names. "
            "Do NOT embed [Source N] labels in this text — citations are "
            "captured separately in the citations field."
        )
    )
    citations: list[Citation] = Field(
        description=(
            "List of citations mapping claims to source labels. "
            "Each citation pairs a specific claim from your answer "
            "with the Source label it came from."
        )
    )


# ─────────────────────────────────────────────────────────────────
# Kingdom Advisor (strategic analysis of 10 kingdom cards)
# ─────────────────────────────────────────────────────────────────

class OpeningStrategy(BaseModel):
    three_four_split: str = Field(
        description="Recommended buys for a 3/4 coin opening split and why"
    )
    five_two_split: str = Field(
        description="Recommended buys for a 5/2 coin opening split and why"
    )


class CardCombo(BaseModel):
    cards: list[str] = Field(
        description="The 2-3 card names involved in this combo"
    )
    explanation: str = Field(
        description="Why this combo is powerful, with specific mechanics"
    )


class KingdomAdvisorResponse(BaseModel):
    opening_strategy: OpeningStrategy = Field(
        description="Recommended opening buys for both 3/4 and 5/2 splits"
    )
    key_combos: list[CardCombo] = Field(
        description="The 2-3 strongest card synergies in this kingdom"
    )
    archetype_assessment: str = Field(
        description=(
            "Is this kingdom suited for Engine, Big Money, Rush, or Slog? "
            "Explain why given these specific cards."
        )
    )
    cards_to_avoid: list[str] = Field(
        description=(
            "Cards that are traps or low-priority in THIS kingdom, "
            "with brief explanation for each (e.g. 'Workshop — no good "
            "cheap targets to gain')"
        )
    )
    attack_and_defense: str = Field(
        description=(
            "Assessment of attack threats and available defenses. "
            "If no attacks, note it will be a solitaire race."
        )
    )


# ─────────────────────────────────────────────────────────────────
# Simulation Analyzer (bot-vs-bot game analysis)
# ─────────────────────────────────────────────────────────────────

class SimulationAnalysisResponse(BaseModel):
    why_winner_wins: str = Field(
        description="What cards/strategy make the difference for the winning bot"
    )
    key_card_interactions: str = Field(
        description="Combos and interactions that matter in this kingdom"
    )
    optimal_opening: str = Field(
        description="What to buy on turns 1-2 based on the simulation data"
    )
    strategic_recommendations: str = Field(
        description="Actionable advice for a human player based on the data"
    )
    surprising_findings: str = Field(
        description="Anything unexpected or counterintuitive in the data"
    )
