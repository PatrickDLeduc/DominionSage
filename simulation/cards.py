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

    # ── Remaining Base Set kingdom cards ──
    "Artisan":      CardDef("Artisan",      6, ("Action",)),
    "Bandit":       CardDef("Bandit",       5, ("Action", "Attack")),
    "Bureaucrat":   CardDef("Bureaucrat",   4, ("Action", "Attack")),
    "Chapel":       CardDef("Chapel",       2, ("Action",)),
    "Council Room": CardDef("Council Room", 5, ("Action",), plus_cards=4, plus_buys=1),
    "Festival":     CardDef("Festival",     5, ("Action",), plus_actions=2, plus_buys=1, plus_coins=2),
    "Gardens":      CardDef("Gardens",      4, ("Victory",)),
    "Harbinger":    CardDef("Harbinger",    3, ("Action",), plus_cards=1, plus_actions=1),
    "Laboratory":   CardDef("Laboratory",   5, ("Action",), plus_cards=2, plus_actions=1),
    "Library":      CardDef("Library",      5, ("Action",)),
    "Moneylender":  CardDef("Moneylender",  4, ("Action",)),
    "Poacher":      CardDef("Poacher",      4, ("Action",), plus_cards=1, plus_actions=1, plus_coins=1),
    "Sentry":       CardDef("Sentry",       5, ("Action",), plus_cards=1, plus_actions=1),
    "Throne Room":  CardDef("Throne Room",  4, ("Action",)),
    "Vassal":       CardDef("Vassal",       3, ("Action",), plus_coins=2),
    "Witch":        CardDef("Witch",        5, ("Action", "Attack"), plus_cards=2),
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
    elif card_name == "Artisan":
        _play_artisan(game, player, bot)
    elif card_name == "Bandit":
        _play_bandit(game, player, bot)
    elif card_name == "Bureaucrat":
        _play_bureaucrat(game, player, bot)
    elif card_name == "Chapel":
        _play_chapel(game, player, bot)
    elif card_name == "Council Room":
        _play_council_room(game, player, bot)
    elif card_name == "Harbinger":
        _play_harbinger(game, player, bot)
    elif card_name == "Library":
        _play_library(game, player, bot)
    elif card_name == "Moneylender":
        _play_moneylender(game, player, bot)
    elif card_name == "Poacher":
        _play_poacher(game, player, bot)
    elif card_name == "Sentry":
        _play_sentry(game, player, bot)
    elif card_name == "Throne Room":
        _play_throne_room(game, player, bot)
    elif card_name == "Vassal":
        _play_vassal(game, player, bot)
    elif card_name == "Witch":
        _play_witch(game, player, bot)
    # Merchant, Market, Smithy, Village, Moat, Festival, Laboratory — handled by standard bonuses


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


def _play_artisan(game: "GameState", player: "PlayerState", bot: "Bot"):
    """Artisan: Gain card costing <= 5 to hand. Put a card from hand to deck."""
    gain_card = bot.choose_artisan_gain(game, player)
    if gain_card:
        card_def = CARD_DEFS.get(gain_card)
        if card_def and card_def.cost <= 5 and game.can_gain(gain_card):
            game.gain_card_to_hand(player, gain_card)
            game.log.append(f"  {player.name} plays Artisan, gains {gain_card} to hand")
            
    if player.hand:
        topdeck_card = bot.choose_artisan_topdeck(game, player)
        if topdeck_card and topdeck_card in player.hand:
            player.hand.remove(topdeck_card)
            player.deck.append(topdeck_card)


def _play_bandit(game: "GameState", player: "PlayerState", bot: "Bot"):
    """Bandit: Gain a Gold. Each other player reveals top 2, trashes non-Copper Treasure, discards rest."""
    if game.can_gain("Gold"):
        game.gain_card(player, "Gold")
        game.log.append(f"  {player.name} plays Bandit, gains Gold")
        
    other = game.other_player
    if "Moat" in other.hand:
        game.log.append(f"  {other.name} reveals Moat — blocked!")
        return

    revealed = other.draw(2)
    treasures = [c for c in revealed if "Treasure" in CARD_DEFS[c].types and c != "Copper"]
    
    if treasures:
        # Choose most expensive treasure to trash
        trash_card = sorted(treasures, key=lambda c: CARD_DEFS[c].cost, reverse=True)[0]
        revealed.remove(trash_card)
        game.trash.append(trash_card)
        game.log.append(f"  {other.name} trashes {trash_card}")
    
    other.discard.extend(revealed)


def _play_bureaucrat(game: "GameState", player: "PlayerState", bot: "Bot"):
    """Bureaucrat: Gain Silver to top of deck. Others reveal/topdeck Victory card."""
    if game.can_gain("Silver"):
        game.supply["Silver"] -= 1
        player.deck.append("Silver")
        game.log.append(f"  {player.name} plays Bureaucrat, gains Silver onto deck")
        
    other = game.other_player
    if "Moat" in other.hand:
        game.log.append(f"  {other.name} reveals Moat — blocked!")
        return
        
    victory_in_hand = [c for c in other.hand if "Victory" in CARD_DEFS[c].types]
    if victory_in_hand:
        topdeck = bot.choose_bureaucrat_victory(game, other, victory_in_hand)
        if topdeck and topdeck in other.hand:
            other.hand.remove(topdeck)
            other.deck.append(topdeck)
            game.log.append(f"  {other.name} puts {topdeck} onto deck")


def _play_chapel(game: "GameState", player: "PlayerState", bot: "Bot"):
    """Chapel: Trash up to 4 cards from hand."""
    trash_cards = bot.choose_chapel_trash(game, player)
    trashed_amount = min(len(trash_cards), 4)
    for c in trash_cards[:trashed_amount]:
        if c in player.hand:
            game.trash_card_from_hand(player, c)
            
    if trashed_amount > 0:
        game.log.append(f"  {player.name} plays Chapel, trashes {trashed_amount} cards")


def _play_council_room(game: "GameState", player: "PlayerState", bot: "Bot"):
    """Council Room: +4 Cards, +1 Buy (done). Each other player draws a card."""
    other = game.other_player
    other.draw(1)


def _play_harbinger(game: "GameState", player: "PlayerState", bot: "Bot"):
    """Harbinger: Look through discard pile. May put a card onto deck."""
    if not player.discard:
        return
    topdeck_card = bot.choose_harbinger_topdeck(game, player)
    if topdeck_card and topdeck_card in player.discard:
        player.discard.remove(topdeck_card)
        player.deck.append(topdeck_card)
        game.log.append(f"  {player.name} plays Harbinger, puts {topdeck_card} onto deck")


def _play_library(game: "GameState", player: "PlayerState", bot: "Bot"):
    """Library: Draw to 7 cards, setting aside Action cards you don't want."""
    set_aside = []
    
    while len(player.hand) < 7:
        drawn = player.draw(1)
        if not drawn:
            break
        
        card = drawn[0]
        if "Action" in CARD_DEFS[card].types:
            keep = bot.choose_library_keep_action(game, player, card)
            if not keep:
                player.hand.remove(card)
                set_aside.append(card)
                
    player.discard.extend(set_aside)


def _play_moneylender(game: "GameState", player: "PlayerState", bot: "Bot"):
    """Moneylender: May trash a Copper from hand for +3 Coins."""
    if "Copper" in player.hand:
        trash = bot.choose_moneylender_trash(game, player)
        if trash:
            game.trash_card_from_hand(player, "Copper")
            player.coins += 3
            game.log.append(f"  {player.name} plays Moneylender, trashes Copper, +3 Coins")


def _play_poacher(game: "GameState", player: "PlayerState", bot: "Bot"):
    """Poacher: Discard a card per empty Supply pile."""
    empty_piles = sum(1 for count in game.supply.values() if count == 0)
    for _ in range(empty_piles):
        if player.hand:
            discard = bot.choose_poacher_discard(game, player)
            if discard and discard in player.hand:
                player.hand.remove(discard)
                player.discard.append(discard)


def _play_sentry(game: "GameState", player: "PlayerState", bot: "Bot"):
    """Sentry: Look at top 2. Trash/discard/topdeck in any order."""
    revealed = player.draw(2)
    # Remove from hand (since draw() puts them in hand)
    for c in revealed:
        if c in player.hand:
            player.hand.remove(c)
            
    for c in list(revealed):
        action = bot.choose_sentry_action(game, player, c)
        if action == "trash":
            game.trash.append(c)
            revealed.remove(c)
        elif action == "discard":
            player.discard.append(c)
            revealed.remove(c)
            
    # Put rest back on deck (bot ordering simplified to just put back)
    player.deck.extend(revealed)


def _play_throne_room(game: "GameState", player: "PlayerState", bot: "Bot"):
    """Throne Room: You may play an Action card from your hand twice."""
    action_in_hand = [c for c in player.hand if "Action" in CARD_DEFS[c].types]
    if action_in_hand:
        card = bot.choose_throne_room_action(game, player, action_in_hand)
        if card and card in player.hand:
            player.hand.remove(card)
            player.play_area.append(card)
            game.log.append(f"  {player.name} plays Throne Room on {card}")
            
            # Play it twice
            for _ in range(2):
                play_card(card, game, player, bot)


def _play_vassal(game: "GameState", player: "PlayerState", bot: "Bot"):
    """Vassal: Discard top card. If Action, may play it."""
    drawn = player.draw(1)
    if not drawn:
        return
        
    card = drawn[0]
    player.hand.remove(card)  # draw() put it in hand
    
    if "Action" in CARD_DEFS[card].types:
        play = bot.choose_vassal_play(game, player, card)
        if play:
            player.play_area.append(card)
            game.log.append(f"  {player.name} plays Vassal, plays {card} from deck")
            play_card(card, game, player, bot)
            return
            
    player.discard.append(card)


def _play_witch(game: "GameState", player: "PlayerState", bot: "Bot"):
    """Witch: Each other player gains a Curse."""
    other = game.other_player
    if "Moat" in other.hand:
        game.log.append(f"  {other.name} reveals Moat — blocked!")
        return
        
    if game.can_gain("Curse"):
        game.gain_card(other, "Curse")
        game.log.append(f"  {player.name} plays Witch, {other.name} gains Curse")
