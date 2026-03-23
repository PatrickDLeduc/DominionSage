"""
simulation/cards.py — Card definitions and effects for the First Game kingdom.

Each card is defined as a CardDef dataclass and has an effect function
that modifies game/player state when played.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation.engine import GameState, PlayerState
    from simulation.bots import Bot


# ─────────────────────────────────────────────────────────────────
# Card definition
# ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CardDef:
    """Static definition of a Dominion card."""
    name: str
    cost: int
    types: tuple[str, ...] = ("Action",)
    coins: int = 0       # coins produced when played (for treasures)
    vp: int = 0          # victory points (for victory cards)
    plus_cards: int = 0
    plus_actions: int = 0
    plus_buys: int = 0
    plus_coins: int = 0


# ─────────────────────────────────────────────────────────────────
# Card definitions
# ─────────────────────────────────────────────────────────────────

CARD_DEFS: dict[str, CardDef] = {
    # ── Basic treasures ──
    "Copper":   CardDef("Copper",   0, ("Treasure",), coins=1),
    "Silver":   CardDef("Silver",   3, ("Treasure",), coins=2),
    "Gold":     CardDef("Gold",     6, ("Treasure",), coins=3),

    # ── Basic victory ──
    "Estate":   CardDef("Estate",   2, ("Victory",), vp=1),
    "Duchy":    CardDef("Duchy",    5, ("Victory",), vp=3),
    "Province": CardDef("Province", 8, ("Victory",), vp=6),

    # ── Curse ──
    "Curse":    CardDef("Curse",    0, ("Curse",), vp=-1),

    # ── First Game kingdom cards ──
    "Cellar":    CardDef("Cellar",    2, ("Action",), plus_actions=1),
    "Market":    CardDef("Market",    5, ("Action",), plus_cards=1, plus_actions=1,
                         plus_buys=1, plus_coins=1),
    "Merchant":  CardDef("Merchant",  3, ("Action",), plus_cards=1, plus_actions=1),
    "Militia":   CardDef("Militia",   4, ("Action", "Attack"), plus_coins=2),
    "Mine":      CardDef("Mine",      5, ("Action",)),
    "Moat":      CardDef("Moat",      2, ("Action", "Reaction"), plus_cards=2),
    "Remodel":   CardDef("Remodel",   4, ("Action",)),
    "Smithy":    CardDef("Smithy",    4, ("Action",), plus_cards=3),
    "Village":   CardDef("Village",   3, ("Action",), plus_cards=1, plus_actions=2),
    "Workshop":  CardDef("Workshop",  3, ("Action",)),
}


# ─────────────────────────────────────────────────────────────────
# Card effects
# ─────────────────────────────────────────────────────────────────

def play_card(card_name: str, game: "GameState", player: "PlayerState",
              bot: "Bot"):
    """Execute a card's effect when played."""
    card = CARD_DEFS.get(card_name)
    if card is None:
        return

    # Apply standard bonuses
    if card.plus_cards > 0:
        player.draw(card.plus_cards)
    if card.plus_actions > 0:
        player.actions += card.plus_actions
    if card.plus_buys > 0:
        player.buys += card.plus_buys
    if card.plus_coins > 0:
        player.coins += card.plus_coins

    # Card-specific effects
    if card_name == "Cellar":
        _play_cellar(game, player, bot)
    elif card_name == "Militia":
        _play_militia(game, player, bot)
    elif card_name == "Mine":
        _play_mine(game, player, bot)
    elif card_name == "Remodel":
        _play_remodel(game, player, bot)
    elif card_name == "Workshop":
        _play_workshop(game, player, bot)
    # Merchant, Market, Smithy, Village, Moat — handled by standard bonuses


def _play_cellar(game: "GameState", player: "PlayerState", bot: "Bot"):
    """Cellar: discard any number of cards, then draw that many."""
    cards_to_discard = bot.choose_cellar_discard(game, player)
    for c in cards_to_discard:
        if c in player.hand:
            player.hand.remove(c)
            player.discard.append(c)
    if cards_to_discard:
        player.draw(len(cards_to_discard))
        game.log.append(f"  {player.name} plays Cellar, discards {len(cards_to_discard)}, draws {len(cards_to_discard)}")


def _play_militia(game: "GameState", player: "PlayerState", bot: "Bot"):
    """Militia: +2 Coins, each other player discards down to 3."""
    other = game.other_player
    # Check for Moat (simplified: if Moat in hand, block)
    if "Moat" in other.hand:
        game.log.append(f"  {other.name} reveals Moat — blocked!")
        return

    while len(other.hand) > 3:
        # For the defending player, use a simple heuristic:
        # discard the least valuable card
        discard_card = _choose_militia_discard(other)
        other.hand.remove(discard_card)
        other.discard.append(discard_card)

    game.log.append(f"  {player.name} plays Militia, {other.name} discards to 3")


def _choose_militia_discard(player: "PlayerState") -> str:
    """Choose what to discard to Militia (simple heuristic)."""
    # Priority: discard Curse > Estate > Copper > other
    priority = ["Curse", "Estate", "Copper"]
    for card_name in priority:
        if card_name in player.hand:
            return card_name
    # Discard cheapest card
    return sorted(player.hand, key=lambda c: CARD_DEFS.get(c, CARD_DEFS["Copper"]).cost)[0]


def _play_mine(game: "GameState", player: "PlayerState", bot: "Bot"):
    """Mine: Trash a Treasure from hand, gain one costing up to +3 to hand."""
    treasures_in_hand = [c for c in player.hand if "Treasure" in CARD_DEFS[c].types]
    if not treasures_in_hand:
        return

    # Simple strategy: upgrade Copper→Silver, Silver→Gold
    trash_card = bot.choose_mine_trash(game, player, treasures_in_hand)
    if trash_card is None:
        return

    trash_cost = CARD_DEFS[trash_card].cost
    max_gain_cost = trash_cost + 3

    # Choose best treasure to gain
    gain_card = None
    for t_name in ["Gold", "Silver", "Copper"]:
        t_def = CARD_DEFS[t_name]
        if t_def.cost <= max_gain_cost and game.can_gain(t_name):
            gain_card = t_name
            break

    if gain_card:
        game.trash_card_from_hand(player, trash_card)
        game.gain_card_to_hand(player, gain_card)
        game.log.append(f"  {player.name} plays Mine: trashes {trash_card}, gains {gain_card} to hand")


def _play_remodel(game: "GameState", player: "PlayerState", bot: "Bot"):
    """Remodel: Trash a card from hand, gain one costing up to +2."""
    if not player.hand:
        return

    trash_card = bot.choose_remodel_trash(game, player)
    if trash_card is None or trash_card not in player.hand:
        return

    trash_cost = CARD_DEFS.get(trash_card, CARD_DEFS["Copper"]).cost
    max_gain_cost = trash_cost + 2

    gain_card = bot.choose_remodel_gain(game, player, max_gain_cost)
    if gain_card is None:
        return

    if CARD_DEFS.get(gain_card, CARD_DEFS["Copper"]).cost <= max_gain_cost and game.can_gain(gain_card):
        game.trash_card_from_hand(player, trash_card)
        game.gain_card(player, gain_card)
        game.log.append(f"  {player.name} plays Remodel: trashes {trash_card}, gains {gain_card}")


def _play_workshop(game: "GameState", player: "PlayerState", bot: "Bot"):
    """Workshop: Gain a card costing up to 4."""
    gain_card = bot.choose_workshop_gain(game, player)
    if gain_card is None:
        return

    card_def = CARD_DEFS.get(gain_card)
    if card_def and card_def.cost <= 4 and game.can_gain(gain_card):
        game.gain_card(player, gain_card)
        game.log.append(f"  {player.name} plays Workshop, gains {gain_card}")
