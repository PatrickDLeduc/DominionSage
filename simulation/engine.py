"""
simulation/engine.py — Simplified Dominion game engine.

Handles game state, turn phases, deck management, and game-over
conditions. Designed for bot-vs-bot simulation on the First Game kingdom.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation.bots import Bot


# ─────────────────────────────────────────────────────────────────
# Player state
# ─────────────────────────────────────────────────────────────────

@dataclass
class PlayerState:
    """Tracks one player's deck, hand, and resources."""

    name: str
    deck: list[str] = field(default_factory=list)
    hand: list[str] = field(default_factory=list)
    discard: list[str] = field(default_factory=list)
    play_area: list[str] = field(default_factory=list)

    # Per-turn resources (reset each turn)
    actions: int = 1
    buys: int = 1
    coins: int = 0

    # Tracking
    merchant_silver_bonus: bool = False  # has Merchant bonus been applied?

    def __post_init__(self):
        """Start with 7 Coppers and 3 Estates, shuffled."""
        if not self.deck:
            self.deck = ["Copper"] * 7 + ["Estate"] * 3
            random.shuffle(self.deck)

    def draw(self, n: int = 1) -> list[str]:
        """Draw n cards from deck. Shuffles discard into deck if needed."""
        drawn = []
        for _ in range(n):
            if not self.deck:
                if not self.discard:
                    break  # nothing left to draw
                self.deck = self.discard[:]
                self.discard.clear()
                random.shuffle(self.deck)
            drawn.append(self.deck.pop())
        self.hand.extend(drawn)
        return drawn

    def draw_hand(self):
        """Draw a fresh hand of 5 cards."""
        self.draw(5)

    def discard_hand_and_play_area(self):
        """Move hand + play area to discard pile (cleanup phase)."""
        self.discard.extend(self.hand)
        self.discard.extend(self.play_area)
        self.hand.clear()
        self.play_area.clear()

    def all_cards(self) -> list[str]:
        """Return every card the player owns (for scoring)."""
        return self.deck + self.hand + self.discard + self.play_area

    def count_in_deck(self, card_name: str) -> int:
        """Count how many copies of a card the player owns total."""
        return self.all_cards().count(card_name)

    def total_cards(self) -> int:
        """Total number of cards owned."""
        return len(self.all_cards())

    def victory_points(self) -> int:
        """Calculate total VP from all owned cards."""
        from simulation.cards import CARD_DEFS
        vp = 0
        all_c = self.all_cards()
        for card_name in all_c:
            card = CARD_DEFS.get(card_name)
            if card and card.vp != 0:
                vp += card.vp
            # Gardens: 1 VP per 10 cards
            if card_name == "Gardens":
                vp += len(all_c) // 10
        return vp

    def reset_turn(self):
        """Reset per-turn resources."""
        self.actions = 1
        self.buys = 1
        self.coins = 0
        self.merchant_silver_bonus = False


# ─────────────────────────────────────────────────────────────────
# Game state
# ─────────────────────────────────────────────────────────────────

@dataclass
class GameState:
    """Tracks the full game: supply piles, players, turn count, log."""

    supply: dict[str, int] = field(default_factory=dict)
    trash: list[str] = field(default_factory=list)
    players: list[PlayerState] = field(default_factory=list)
    turn_number: int = 0
    current_player_idx: int = 0
    log: list[str] = field(default_factory=list)
    game_over: bool = False

    @property
    def current_player(self) -> PlayerState:
        return self.players[self.current_player_idx]

    def get_other_players(self) -> list[PlayerState]:
        """Returns the other players in turn order (starting from next player)."""
        idx = self.current_player_idx
        return self.players[idx + 1:] + self.players[:idx]

    def can_gain(self, card_name: str) -> bool:
        """Check if a card is available in the supply."""
        return self.supply.get(card_name, 0) > 0

    def gain_card(self, player: PlayerState, card_name: str):
        """Move a card from supply to player's discard pile."""
        if self.supply.get(card_name, 0) > 0:
            self.supply[card_name] -= 1
            player.discard.append(card_name)

    def gain_card_to_hand(self, player: PlayerState, card_name: str):
        """Move a card from supply to player's hand."""
        if self.supply.get(card_name, 0) > 0:
            self.supply[card_name] -= 1
            player.hand.append(card_name)

    def trash_card_from_hand(self, player: PlayerState, card_name: str):
        """Remove a card from player's hand and put it in the trash."""
        if card_name in player.hand:
            player.hand.remove(card_name)
            self.trash.append(card_name)

    def check_game_over(self) -> bool:
        """Game ends if Province pile is empty or any 3 piles are empty."""
        if self.supply.get("Province", 0) == 0:
            self.game_over = True
            return True
        empty_piles = sum(1 for count in self.supply.values() if count == 0)
        if empty_piles >= 3:
            self.game_over = True
            return True
        return False


# ─────────────────────────────────────────────────────────────────
# Game setup
# ─────────────────────────────────────────────────────────────────

def setup_game(kingdom_cards: list[str], player_names: list[str] = None) -> GameState:
    """
    Initialize a game with the given kingdom cards.
    Sets up supply piles with correct counts based on the number of players.
    """
    if player_names is None:
        player_names = ["Player 1", "Player 2"]
        
    num_players = len(player_names)
    assert 2 <= num_players <= 4, "Game only supports 2-4 players"

    # Basic victory scaling: 8 for 2 players, 12 for 3 or 4 players
    victory_count = 8 if num_players == 2 else 12
    # Curses: 10 per player beyond the first
    curse_count = (num_players - 1) * 10

    # Basic supply
    supply = {
        "Copper": 60 - (7 * num_players),
        "Silver": 40,
        "Gold": 30,
        "Estate": victory_count,
        "Duchy": victory_count,
        "Province": victory_count,
        "Curse": curse_count,
    }

    # Kingdom piles (10 each; Victory kingdom cards get standard victory_count)
    from simulation.cards import CARD_DEFS
    for card_name in kingdom_cards:
        card = CARD_DEFS.get(card_name)
        if card and "Victory" in card.types:
            supply[card_name] = victory_count
        else:
            supply[card_name] = 10

    # Create players
    players = [PlayerState(name=name) for name in player_names]

    # Draw initial hands
    for p in players:
        p.draw_hand()

    game = GameState(supply=supply, players=players)
    return game


# ─────────────────────────────────────────────────────────────────
# Turn execution
# ─────────────────────────────────────────────────────────────────

def play_turn(game: GameState, bot: "Bot"):
    """
    Execute one full turn for the current player using the bot's strategy.

    Phases: Action → Buy → Cleanup
    """
    player = game.current_player

    # Reset per-turn resources
    player.reset_turn()

    # ── Action Phase ────────────────────────────────────────────
    while player.actions > 0:
        action_card = bot.choose_action(game, player)
        if action_card is None:
            break
        if action_card not in player.hand:
            break

        # Play the action card
        player.hand.remove(action_card)
        player.play_area.append(action_card)
        player.actions -= 1

        # Execute card effect
        from simulation.cards import play_card
        play_card(action_card, game, player, bot)

    # ── Buy Phase (play treasures first) ────────────────────────
    play_treasures(game, player)

    while player.buys > 0:
        card_to_buy = bot.choose_buy(game, player)
        if card_to_buy is None:
            break

        from simulation.cards import CARD_DEFS
        card = CARD_DEFS.get(card_to_buy)
        if card is None:
            break
        if player.coins < card.cost:
            break
        if not game.can_gain(card_to_buy):
            break

        player.coins -= card.cost
        player.buys -= 1
        game.gain_card(player, card_to_buy)
        game.log.append(f"  {player.name} buys {card_to_buy}")

    # ── Cleanup Phase ───────────────────────────────────────────
    player.discard_hand_and_play_area()
    player.draw_hand()

    # Check game end
    game.check_game_over()


def play_treasures(game: GameState, player: PlayerState):
    """Play all treasure cards from hand, adding coins."""
    from simulation.cards import CARD_DEFS

    treasures = [c for c in player.hand if "Treasure" in CARD_DEFS.get(c, CARD_DEFS["Copper"]).types]

    # Play Merchant bonus tracking: play non-Silver treasures first,
    # then Silver (so Merchant bonus can be checked)
    non_silver = [t for t in treasures if t != "Silver"]
    silvers = [t for t in treasures if t == "Silver"]

    for t in non_silver + silvers:
        player.hand.remove(t)
        player.play_area.append(t)
        card = CARD_DEFS[t]
        player.coins += card.coins

        # Merchant bonus: +1 coin on first Silver played
        if t == "Silver" and not player.merchant_silver_bonus:
            merchant_count = player.play_area.count("Merchant")
            if merchant_count > 0:
                player.coins += merchant_count
                player.merchant_silver_bonus = True


# ─────────────────────────────────────────────────────────────────
# Full game loop
# ─────────────────────────────────────────────────────────────────

def run_game(kingdom_cards: list[str], bots: list["Bot"],
             player_names: list[str] = None, max_turns: int = 100) -> dict:
    """
    Play a full game and return results.

    Returns dict with:
      - winner: name of winner (or "Tie")
      - vp: {name: vp} for each player
      - turns: total turns played
      - log: full game log
      - buy_counts: {name: {card: count}} purchases per player
    """
    game = setup_game(kingdom_cards, player_names)
    buy_counts = {p.name: {} for p in game.players}

    total_turns = 0

    while not game.game_over and total_turns < max_turns * len(bots):
        # turn_number corresponds to round number basically
        game.turn_number = total_turns // len(bots) + 1
        player = game.current_player
        bot = bots[game.current_player_idx]

        game.log.append(f"Turn {game.turn_number} — {player.name}")

        # Track buys
        pre_discard = len(player.discard)
        play_turn(game, bot)
        # Count new cards gained (buys)
        new_cards = player.discard[pre_discard:] + \
                    [c for c in player.hand if c not in ["Copper", "Silver", "Gold",
                     "Estate", "Duchy", "Province", "Curse"]]

        for line in game.log:
            if line.startswith(f"  {player.name} buys "):
                card_name = line.replace(f"  {player.name} buys ", "")
                buy_counts[player.name][card_name] = \
                    buy_counts[player.name].get(card_name, 0) + 1

        # Next player
        game.current_player_idx = (game.current_player_idx + 1) % len(bots)
        total_turns += 1

    # Score
    scores = {p.name: p.victory_points() for p in game.players}

    # Determine winner (ties broken by fewer turns — but both play same # of turns)
    max_vp = max(scores.values())
    winners = [name for name, vp in scores.items() if vp == max_vp]
    winner = winners[0] if len(winners) == 1 else "Tie"

    return {
        "winner": winner,
        "vp": scores,
        "turns": game.turn_number,
        "log": game.log,
        "buy_counts": buy_counts,
    }
