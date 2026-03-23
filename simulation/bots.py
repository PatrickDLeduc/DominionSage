"""
simulation/bots.py — Bot strategies for Dominion simulation.

Each bot implements decision methods for action phase, buy phase,
and card-specific choices (Cellar discard, Remodel targets, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation.engine import GameState, PlayerState


class Bot(ABC):
    """Base class for Dominion bots."""

    name: str = "Bot"

    @abstractmethod
    def choose_action(self, game: "GameState", player: "PlayerState") -> str | None:
        """Choose an Action card to play, or None to stop."""

    @abstractmethod
    def choose_buy(self, game: "GameState", player: "PlayerState") -> str | None:
        """Choose a card to buy, or None to stop buying."""

    def choose_cellar_discard(self, game: "GameState", player: "PlayerState") -> list[str]:
        """Choose cards to discard with Cellar. Default: discard victory cards."""
        return [c for c in player.hand if c in ("Estate", "Duchy", "Curse")]

    def choose_mine_trash(self, game: "GameState", player: "PlayerState",
                          treasures: list[str]) -> str | None:
        """Choose a Treasure to trash with Mine. Default: upgrade cheapest."""
        if "Copper" in treasures and game.can_gain("Silver"):
            return "Copper"
        if "Silver" in treasures and game.can_gain("Gold"):
            return "Silver"
        return None

    def choose_remodel_trash(self, game: "GameState", player: "PlayerState") -> str | None:
        """Choose a card to trash with Remodel. Default: trash Estate→something."""
        if "Estate" in player.hand:
            return "Estate"
        if "Copper" in player.hand:
            return "Copper"
        return None

    def choose_remodel_gain(self, game: "GameState", player: "PlayerState",
                            max_cost: int) -> str | None:
        """Choose a card to gain with Remodel."""
        from simulation.cards import CARD_DEFS
        # Gain the best card we can afford
        priority = ["Province", "Gold", "Duchy", "Silver", "Market",
                     "Smithy", "Militia", "Village"]
        for card in priority:
            cd = CARD_DEFS.get(card)
            if cd and cd.cost <= max_cost and game.can_gain(card):
                return card
        return None

    def choose_workshop_gain(self, game: "GameState", player: "PlayerState") -> str | None:
        """Choose a card to gain with Workshop (cost ≤ 4)."""
        return "Silver"  # safe default


# ─────────────────────────────────────────────────────────────────
# Bot 1: Big Money + Smithy
# ─────────────────────────────────────────────────────────────────

class BigMoneyBot(Bot):
    """
    Classic Big Money strategy with light Smithy support.

    Buy priority: Province > Gold > Smithy (max 2) > Silver
    Only plays Smithy/Market/Merchant/Village for draw; skips most other actions.
    """

    name = "Big Money"

    def choose_action(self, game: "GameState", player: "PlayerState") -> str | None:
        # Play cantrips and draw cards, but be conservative
        action_priority = ["Village", "Market", "Merchant", "Smithy",
                           "Cellar", "Mine", "Moat"]
        for card in action_priority:
            if card in player.hand:
                return card
        return None

    def choose_buy(self, game: "GameState", player: "PlayerState") -> str | None:
        coins = player.coins

        # Late game: buy Duchy when Provinces are running low
        provinces_left = game.supply.get("Province", 0)

        if coins >= 8 and game.can_gain("Province"):
            return "Province"
        if coins >= 6 and game.can_gain("Gold"):
            return "Gold"
        # Buy at most 2 Smithies
        if coins >= 4 and game.can_gain("Smithy") and player.count_in_deck("Smithy") < 2:
            return "Smithy"
        if coins >= 5 and provinces_left <= 4 and game.can_gain("Duchy"):
            return "Duchy"
        if coins >= 3 and game.can_gain("Silver"):
            return "Silver"
        return None


# ─────────────────────────────────────────────────────────────────
# Bot 2: Village / Smithy Engine
# ─────────────────────────────────────────────────────────────────

class EngineBot(Bot):
    """
    Village/Smithy engine strategy.

    Builds an action-dense deck with Villages (for actions) and Smithies
    (for draw), adds Markets for economy, then transitions to buying
    Provinces once the engine can consistently produce 8+ coins.
    """

    name = "Engine"

    def choose_action(self, game: "GameState", player: "PlayerState") -> str | None:
        # Play Villages first (for +Actions), then draw cards
        action_priority = ["Village", "Market", "Cellar", "Merchant",
                           "Smithy", "Workshop", "Remodel", "Mine",
                           "Militia", "Moat"]
        for card in action_priority:
            if card in player.hand:
                return card
        return None

    def choose_buy(self, game: "GameState", player: "PlayerState") -> str | None:
        coins = player.coins
        provinces_left = game.supply.get("Province", 0)
        total_cards = player.total_cards()
        villages = player.count_in_deck("Village")
        smithies = player.count_in_deck("Smithy")
        markets = player.count_in_deck("Market")

        # Always buy Province if possible
        if coins >= 8 and game.can_gain("Province"):
            return "Province"

        # Late game green
        if coins >= 5 and provinces_left <= 3 and game.can_gain("Duchy"):
            return "Duchy"

        # Engine building phase
        if total_cards < 20:
            # Prioritize engine components
            if coins >= 5 and game.can_gain("Market") and markets < 3:
                return "Market"
            if coins >= 4 and game.can_gain("Smithy") and smithies < villages + 1 and smithies < 3:
                return "Smithy"
            if coins >= 3 and game.can_gain("Village") and villages < smithies + 2 and villages < 4:
                return "Village"
            if coins >= 3 and game.can_gain("Merchant") and player.count_in_deck("Merchant") < 2:
                return "Merchant"

        # Transition: buy Gold when engine is assembled
        if coins >= 6 and game.can_gain("Gold"):
            return "Gold"

        # Fill gaps
        if coins >= 3 and game.can_gain("Silver"):
            return "Silver"

        return None

    def choose_workshop_gain(self, game: "GameState", player: "PlayerState") -> str | None:
        """Workshop: gain engine components."""
        villages = player.count_in_deck("Village")
        smithies = player.count_in_deck("Smithy")

        if villages < smithies + 1 and game.can_gain("Village"):
            return "Village"
        if smithies < 3 and game.can_gain("Smithy"):
            return "Smithy"
        if game.can_gain("Silver"):
            return "Silver"
        return None

    def choose_cellar_discard(self, game: "GameState", player: "PlayerState") -> list[str]:
        """Cellar: discard victory cards and excess Coppers."""
        to_discard = []
        for c in player.hand:
            if c in ("Estate", "Duchy", "Curse"):
                to_discard.append(c)
            elif c == "Copper" and player.hand.count("Copper") > 2:
                to_discard.append(c)
        return to_discard

    def choose_remodel_trash(self, game: "GameState", player: "PlayerState") -> str | None:
        """Remodel: upgrade Estates early, later Golds→Provinces."""
        provinces_left = game.supply.get("Province", 0)
        if "Gold" in player.hand and provinces_left > 0 and game.can_gain("Province"):
            return "Gold"
        if "Estate" in player.hand:
            return "Estate"
        if "Copper" in player.hand:
            return "Copper"
        return None

    def choose_remodel_gain(self, game: "GameState", player: "PlayerState",
                            max_cost: int) -> str | None:
        """Remodel gain: Province if possible, otherwise engine pieces."""
        from simulation.cards import CARD_DEFS
        priority = ["Province", "Gold", "Market", "Smithy", "Village", "Silver"]
        for card in priority:
            cd = CARD_DEFS.get(card)
            if cd and cd.cost <= max_cost and game.can_gain(card):
                return card
        return None
