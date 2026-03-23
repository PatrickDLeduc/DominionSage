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

    def choose_artisan_gain(self, game: "GameState", player: "PlayerState") -> str | None:
        """Choose a card to gain with Artisan (cost ≤ 5)."""
        return "Market" if game.can_gain("Market") else "Silver"

    def choose_artisan_topdeck(self, game: "GameState", player: "PlayerState") -> str | None:
        """Choose a card from hand to put onto deck for Artisan."""
        # Topdeck the card we just gained if possible, or an Action
        for c in ["Market", "Silver", "Copper"]:
            if c in player.hand:
                return c
        return player.hand[0] if player.hand else None

    def choose_bureaucrat_victory(self, game: "GameState", player: "PlayerState",
                                  victory_in_hand: list[str]) -> str | None:
        """Choose a Victory card from hand to topdeck for Bureaucrat."""
        # Priority: Estate > Duchy > Province (give opponent worst card back)
        for c in ["Estate", "Duchy", "Province"]:
            if c in victory_in_hand:
                return c
        return victory_in_hand[0] if victory_in_hand else None

    def choose_chapel_trash(self, game: "GameState", player: "PlayerState") -> list[str]:
        """Choose up to 4 cards to trash with Chapel."""
        to_trash = []
        for c in player.hand:
            if c in ("Curse", "Estate") or (c == "Copper" and player.coins >= 3):
                to_trash.append(c)
        return to_trash

    def choose_harbinger_topdeck(self, game: "GameState", player: "PlayerState") -> str | None:
        """Choose a card from discard to topdeck with Harbinger."""
        from simulation.cards import CARD_DEFS
        # Topdeck best action or treasure
        priority = ["Province", "Gold", "Market", "Smithy", "Silver"]
        for c in priority:
            if c in player.discard:
                return c
        # Otherwise topdeck best action
        actions = [c for c in player.discard if "Action" in CARD_DEFS[c].types]
        if actions:
            return sorted(actions, key=lambda x: CARD_DEFS[x].cost, reverse=True)[0]
        return None

    def choose_library_keep_action(self, game: "GameState", player: "PlayerState", card: str) -> bool:
        """Library: return True to keep an Action card, False to set it aside."""
        from simulation.cards import CARD_DEFS
        # Keep it if it gives actions (so we can play it) or we have actions left
        card_def = CARD_DEFS.get(card)
        if card_def and card_def.plus_actions > 0:
            return True
        if player.actions > 0:
            return True
        return False

    def choose_moneylender_trash(self, game: "GameState", player: "PlayerState") -> bool:
        """Return True to trash a Copper for +3 Coins."""
        return True

    def choose_poacher_discard(self, game: "GameState", player: "PlayerState") -> str | None:
        """Choose a card to discard for Poacher."""
        # Discard worst cards
        priority = ["Curse", "Estate", "Copper", "Duchy", "Silver"]
        for c in priority:
            if c in player.hand:
                return c
        return player.hand[0] if player.hand else None

    def choose_sentry_action(self, game: "GameState", player: "PlayerState", card: str) -> str:
        """Sentry: return 'trash', 'discard', or 'topdeck'."""
        if card in ("Curse", "Estate", "Copper"):
            return "trash"
        if card == "Duchy":
            return "discard"
        return "topdeck"

    def choose_throne_room_action(self, game: "GameState", player: "PlayerState",
                                  actions_in_hand: list[str]) -> str | None:
        """Choose an Action card to play twice with Throne Room."""
        from simulation.cards import CARD_DEFS
        # Throne the most expensive action
        if not actions_in_hand:
            return None
        return sorted(actions_in_hand, key=lambda c: CARD_DEFS[c].cost, reverse=True)[0]

    def choose_vassal_play(self, game: "GameState", player: "PlayerState", card: str) -> bool:
        """Vassal: return True to play the discarded Action card."""
        return True


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
        action_priority = ["Laboratory", "Festival", "Village", "Market", "Merchant",
                           "Smithy", "Council Room", "Witch", "Sentry", "Poacher",
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
            
        # Optional terminal actions (max 1-2 per deck)
        if coins >= 5 and game.can_gain("Witch") and player.count_in_deck("Witch") < 1:
            return "Witch"
        if coins >= 5 and game.can_gain("Laboratory") and player.count_in_deck("Laboratory") < 2:
            return "Laboratory"
        if coins >= 5 and game.can_gain("Council Room") and player.count_in_deck("Council Room") < 1:
            return "Council Room"
            
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
        # Play Villages first (for +Actions), then draw cards, then terminal actions
        action_priority = ["Chapel", "Laboratory", "Village", "Festival", "Throne Room",
                           "Market", "Sentry", "Cellar", "Merchant", "Poacher",
                           "Smithy", "Council Room", "Library", "Witch", "Moneylender",
                           "Workshop", "Remodel", "Mine", "Militia", "Moat", "Bandit"]
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
            # Chapel is top priority for engines early
            if coins >= 2 and total_cards < 12 and game.can_gain("Chapel") and player.count_in_deck("Chapel") < 1:
                return "Chapel"
                
            # Prioritize engine components
            if coins >= 5 and game.can_gain("Laboratory") and player.count_in_deck("Laboratory") < 3:
                return "Laboratory"
            if coins >= 5 and game.can_gain("Market") and markets < 3:
                return "Market"
            if coins >= 5 and game.can_gain("Festival") and player.count_in_deck("Festival") < 2:
                return "Festival"
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


# ─────────────────────────────────────────────────────────────────
# Bot 3: Attacker
# ─────────────────────────────────────────────────────────────────

class AttackerBot(Bot):
    """
    Focuses entirely on disrupting the opponent. Buys Witch, Militia, Bandit early.
    """
    name = "Attacker"

    def choose_action(self, game: "GameState", player: "PlayerState") -> str | None:
        # Play attacks first
        action_priority = ["Witch", "Militia", "Bandit", "Laboratory", "Festival",
                           "Village", "Market", "Smithy", "Moat"]
        for card in action_priority:
            if card in player.hand:
                return card
        return None

    def choose_buy(self, game: "GameState", player: "PlayerState") -> str | None:
        coins = player.coins
        provinces_left = game.supply.get("Province", 0)

        if coins >= 8 and game.can_gain("Province"):
            return "Province"
            
        # Get attacks as top priority
        if coins >= 5 and game.can_gain("Witch") and player.count_in_deck("Witch") < 2:
            return "Witch"
        if coins >= 5 and game.can_gain("Bandit") and player.count_in_deck("Bandit") < 2:
            return "Bandit"
        if coins >= 4 and game.can_gain("Militia") and player.count_in_deck("Militia") < 2:
            return "Militia"
            
        if coins >= 6 and game.can_gain("Gold"):
            return "Gold"
        if coins >= 5 and provinces_left <= 4 and game.can_gain("Duchy"):
            return "Duchy"
        if coins >= 3 and game.can_gain("Silver"):
            return "Silver"
        return None


# ─────────────────────────────────────────────────────────────────
# Bot 4: TrashBot (Thin Deck)
# ─────────────────────────────────────────────────────────────────

class TrashBot(Bot):
    """
    Obsessively thins its deck. Buys Chapel/Sentry/Moneylender and destroys starting cards.
    """
    name = "TrashBot"

    def choose_action(self, game: "GameState", player: "PlayerState") -> str | None:
        # Trashers first
        action_priority = ["Chapel", "Sentry", "Moneylender", "Laboratory", 
                           "Village", "Market", "Smithy"]
        for card in action_priority:
            if card in player.hand:
                return card
        return None

    def choose_buy(self, game: "GameState", player: "PlayerState") -> str | None:
        coins = player.coins
        total_cards = player.total_cards()
        
        # Priority 1: Get a trasher immediately
        if coins >= 5 and game.can_gain("Sentry") and player.count_in_deck("Sentry") < 1:
            return "Sentry"
        if coins >= 4 and game.can_gain("Moneylender") and player.count_in_deck("Moneylender") < 1:
            return "Moneylender"
        if coins >= 2 and game.can_gain("Chapel") and player.count_in_deck("Chapel") < 1:
            return "Chapel"
            
        # Refuse to buy victory cards if deck is still full of junk
        bad_cards = player.count_in_deck("Copper") + player.count_in_deck("Estate") + player.count_in_deck("Curse")
        deck_is_clean = bad_cards <= 2
        
        if coins >= 8 and game.can_gain("Province") and deck_is_clean:
            return "Province"
        if coins >= 6 and game.can_gain("Gold"):
            return "Gold"
        if coins >= 5 and game.supply.get("Province", 0) <= 3 and game.can_gain("Duchy") and deck_is_clean:
            return "Duchy"
            
        # Only buy silver if we have a trasher to get rid of coppers
        has_trasher = player.count_in_deck("Chapel") > 0 or player.count_in_deck("Moneylender") > 0 or player.count_in_deck("Sentry") > 0
        if coins >= 3 and game.can_gain("Silver") and has_trasher:
            return "Silver"
            
        return None


# ─────────────────────────────────────────────────────────────────
# Bot 5: Rusher
# ─────────────────────────────────────────────────────────────────

class RusherBot(Bot):
    """
    Tries to end the game on 3 empty piles. Avoids Provinces, targets Duchies/Gardens 
    and actively finishes off cheap piles.
    """
    name = "Rusher"

    def choose_action(self, game: "GameState", player: "PlayerState") -> str | None:
        action_priority = ["Festival", "Workshop", "Village", "Market", "Smithy"]
        for card in action_priority:
            if card in player.hand:
                return card
        return None

    def choose_buy(self, game: "GameState", player: "PlayerState") -> str | None:
        coins = player.coins
        
        # Find piles that are close to empty (<= 3 cards)
        low_piles = [pile for pile, count in game.supply.items() if count > 0 and count <= 4]
        
        # Rush gardens/duchy
        if coins >= 5 and game.can_gain("Duchy"):
            return "Duchy"
        if coins >= 4 and game.can_gain("Gardens"):
            return "Gardens"
            
        # If we have spare buys, buy out low piles to force end game
        for pile in low_piles:
            from simulation.cards import CARD_DEFS
            cd = CARD_DEFS.get(pile)
            if cd and coins >= cd.cost and game.can_gain(pile):
                return pile
                
        # Otherwise get cheap stuff to boost Gardens/Deck size
        if coins >= 3 and game.can_gain("Workshop") and player.count_in_deck("Workshop") < 3:
            return "Workshop"
        if coins >= 3 and game.can_gain("Silver"):
            return "Silver"
        if game.can_gain("Copper"):
            return "Copper"
            
        return None


# ─────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────

AVAILABLE_BOTS = {
    "Big Money": BigMoneyBot,
    "Engine": EngineBot,
    "Attacker": AttackerBot,
    "TrashBot": TrashBot,
    "Rusher": RusherBot,
}
