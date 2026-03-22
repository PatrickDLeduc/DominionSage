"""
data/kingdom_presets.py — Official recommended kingdom sets from Dominion rulebooks.

Each preset is a dict with:
  - name:       Display name (e.g. "First Game")
  - expansions: List of expansion names involved
  - cards:      List of 10 kingdom card names
"""

KINGDOM_PRESETS = [
    # ── Dominion alone ──────────────────────────────────────────────
    {
        "name": "First Game",
        "expansions": ["Dominion"],
        "cards": ["Cellar", "Market", "Merchant", "Militia", "Mine",
                  "Moat", "Remodel", "Smithy", "Village", "Workshop"],
    },
    {
        "name": "Size Distortion",
        "expansions": ["Dominion"],
        "cards": ["Artisan", "Bandit", "Bureaucrat", "Chapel", "Festival",
                  "Gardens", "Sentry", "Throne Room", "Witch", "Workshop"],
    },
    {
        "name": "Deck Top",
        "expansions": ["Dominion"],
        "cards": ["Artisan", "Bureaucrat", "Council Room", "Festival",
                  "Harbinger", "Laboratory", "Moneylender", "Sentry",
                  "Vassal", "Village"],
    },
    {
        "name": "Sleight of Hand",
        "expansions": ["Dominion"],
        "cards": ["Cellar", "Council Room", "Festival", "Gardens", "Library",
                  "Harbinger", "Militia", "Poacher", "Smithy", "Throne Room"],
    },
    {
        "name": "Improvements",
        "expansions": ["Dominion"],
        "cards": ["Artisan", "Cellar", "Market", "Merchant", "Mine",
                  "Moat", "Moneylender", "Poacher", "Remodel", "Witch"],
    },
    {
        "name": "Silver & Gold",
        "expansions": ["Dominion"],
        "cards": ["Bandit", "Bureaucrat", "Chapel", "Harbinger", "Laboratory",
                  "Merchant", "Mine", "Moneylender", "Throne Room", "Vassal"],
    },

    # ── Dominion & Intrigue ──────────────────────────────────────────
    {
        "name": "Underlings",
        "expansions": ["Dominion", "Intrigue"],
        "cards": ["Cellar", "Festival", "Library", "Sentry", "Vassal",
                  "Courtier", "Diplomat", "Minion", "Nobles", "Pawn"],
    },
    {
        "name": "Grand Scheme",
        "expansions": ["Dominion", "Intrigue"],
        "cards": ["Artisan", "Council Room", "Market", "Militia", "Workshop",
                  "Bridge", "Mill", "Mining Village", "Patrol", "Shanty Town"],
    },
    {
        "name": "Deconstruction",
        "expansions": ["Dominion", "Intrigue"],
        "cards": ["Bandit", "Mine", "Remodel", "Throne Room", "Village",
                  "Diplomat", "Harem", "Lurker", "Replace", "Swindler"],
    },

    # ── Dominion & Seaside ───────────────────────────────────────────
    {
        "name": "Reach for Tomorrow",
        "expansions": ["Dominion", "Seaside"],
        "cards": ["Artisan", "Cellar", "Council Room", "Vassal", "Village",
                  "Cutpurse", "Lookout", "Sea Witch", "Monkey", "Treasure Map"],
    },
    {
        "name": "Repetition",
        "expansions": ["Dominion", "Seaside"],
        "cards": ["Festival", "Harbinger", "Militia", "Workshop",
                  "Caravan", "Cutpurse", "Pirate", "Salvager",
                  "Treasury", "Outpost"],
    },
    {
        "name": "Give and Take",
        "expansions": ["Dominion", "Seaside"],
        "cards": ["Library", "Market", "Moneylender", "Witch",
                  "Fishing Village", "Haven", "Salvager",
                  "Smugglers", "Warehouse", "Wharf"],
    },

    # ── Dominion & Alchemy ───────────────────────────────────────────
    {
        "name": "Forbidden Arts",
        "expansions": ["Dominion", "Alchemy"],
        "cards": ["Bandit", "Cellar", "Council Room", "Gardens",
                  "Laboratory", "Throne Room", "Apprentice", "Familiar",
                  "Possession", "University"],
    },
    {
        "name": "Potion Mixers",
        "expansions": ["Dominion", "Alchemy"],
        "cards": ["Cellar", "Festival", "Militia", "Poacher", "Smithy",
                  "Alchemist", "Apothecary", "Golem", "Herbalist", "Transmute"],
    },
    {
        "name": "Chemistry Lesson",
        "expansions": ["Dominion", "Alchemy"],
        "cards": ["Bureaucrat", "Market", "Moat", "Remodel", "Vassal",
                  "Witch", "Alchemist", "Golem", "Philosopher's Stone",
                  "University"],
    },

    # ── Dominion & Prosperity ────────────────────────────────────────
    {
        "name": "Biggest Money",
        "expansions": ["Dominion", "Prosperity"],
        "cards": ["Artisan", "Harbinger", "Laboratory", "Mine", "Moneylender",
                  "Bank", "Grand Market", "Mint", "Royal Seal", "Venture"],
    },
    {
        "name": "The King's Army",
        "expansions": ["Dominion", "Prosperity"],
        "cards": ["Bureaucrat", "Council Room", "Merchant", "Moat", "Village",
                  "Expand", "Goons", "King's Court", "Rabble", "Vault"],
    },
    {
        "name": "The Good Life",
        "expansions": ["Dominion", "Prosperity"],
        "cards": ["Artisan", "Bureaucrat", "Cellar", "Gardens", "Village",
                  "Contraband", "Counting House", "Hoard", "Monument",
                  "Mountebank"],
    },

    # ── Dominion & Cornucopia/Guilds ─────────────────────────────────
    {
        "name": "Bounty of the Hunt",
        "expansions": ["Dominion", "Cornucopia"],
        "cards": ["Cellar", "Festival", "Militia", "Moneylender", "Smithy",
                  "Harvest", "Horn of Plenty", "Hunting Party", "Menagerie",
                  "Tournament"],
    },
    {
        "name": "Bad Omens",
        "expansions": ["Dominion", "Cornucopia"],
        "cards": ["Bureaucrat", "Laboratory", "Merchant", "Poacher",
                  "Throne Room", "Fortune Teller", "Hamlet",
                  "Horn of Plenty", "Jester", "Remake"],
    },
    {
        "name": "The Jester's Workshop",
        "expansions": ["Dominion", "Cornucopia"],
        "cards": ["Artisan", "Laboratory", "Market", "Remodel", "Workshop",
                  "Fairgrounds", "Farming Village", "Horse Traders",
                  "Jester", "Young Witch"],
    },
    {
        "name": "Arts and Crafts",
        "expansions": ["Dominion", "Guilds"],
        "cards": ["Laboratory", "Cellar", "Workshop", "Festival",
                  "Moneylender", "Stonemason", "Advisor", "Baker",
                  "Journeyman", "Merchant Guild"],
    },
    {
        "name": "Clean Living",
        "expansions": ["Dominion", "Guilds"],
        "cards": ["Bandit", "Militia", "Moneylender", "Gardens", "Village",
                  "Butcher", "Baker", "Candlestick Maker", "Doctor",
                  "Soothsayer"],
    },
    {
        "name": "Gilding the Lily",
        "expansions": ["Dominion", "Guilds"],
        "cards": ["Library", "Merchant", "Remodel", "Market", "Sentry",
                  "Plaza", "Masterpiece", "Candlestick Maker", "Taxman",
                  "Herald"],
    },

    # ── Dominion & Hinterlands ───────────────────────────────────────
    {
        "name": "Highway Robbery",
        "expansions": ["Dominion", "Hinterlands"],
        "cards": ["Cellar", "Library", "Moneylender", "Throne Room",
                  "Workshop", "Highway", "Inn", "Margrave",
                  "Nomads", "Oasis"],
    },
    {
        "name": "Adventures Abroad",
        "expansions": ["Dominion", "Hinterlands"],
        "cards": ["Festival", "Laboratory", "Remodel", "Sentry", "Vassal",
                  "Crossroads", "Farmland", "Fool's Gold",
                  "Souk", "Spice Merchant"],
    },

    # ── Dominion & Dark Ages ─────────────────────────────────────────
    {
        "name": "High and Low",
        "expansions": ["Dominion", "Dark Ages"],
        "cards": ["Cellar", "Moneylender", "Throne Room", "Witch",
                  "Workshop", "Hermit", "Hunting Grounds", "Mystic",
                  "Poor House", "Wandering Minstrel"],
    },
    {
        "name": "Chivalry and Revelry",
        "expansions": ["Dominion", "Dark Ages"],
        "cards": ["Festival", "Gardens", "Laboratory", "Library", "Remodel",
                  "Altar", "Knights", "Rats", "Scavenger", "Squire"],
    },

    # ── Dominion & Adventures ────────────────────────────────────────
    {
        "name": "Level Up",
        "expansions": ["Dominion", "Adventures"],
        "cards": ["Market", "Merchant", "Militia", "Throne Room", "Workshop",
                  "Dungeon", "Gear", "Guide", "Lost City", "Miser"],
    },
    {
        "name": "Son of Size Distortion",
        "expansions": ["Dominion", "Adventures"],
        "cards": ["Bandit", "Bureaucrat", "Gardens", "Moneylender", "Witch",
                  "Amulet", "Duplicate", "Giant", "Messenger",
                  "Treasure Trove"],
    },

    # ── Dominion & Empires ───────────────────────────────────────────
    {
        "name": "Everything in Moderation",
        "expansions": ["Dominion", "Empires"],
        "cards": ["Cellar", "Library", "Remodel", "Village", "Workshop",
                  "Enchantress", "Forum", "Legionary", "Overlord", "Temple"],
    },
    {
        "name": "Silver Bullets",
        "expansions": ["Dominion", "Empires"],
        "cards": ["Bureaucrat", "Gardens", "Laboratory", "Market",
                  "Moneylender", "Catapult", "Charm", "Farmers' Market",
                  "Groundskeeper", "Patrician"],
    },

    # ── Dominion & Nocturne ──────────────────────────────────────────
    {
        "name": "Night Shift",
        "expansions": ["Dominion", "Nocturne"],
        "cards": ["Bandit", "Gardens", "Mine", "Poacher", "Smithy",
                  "Druid", "Exorcist", "Ghost Town", "Idol",
                  "Night Watchman"],
    },
    {
        "name": "Idle Hands",
        "expansions": ["Dominion", "Nocturne"],
        "cards": ["Cellar", "Harbinger", "Market", "Merchant", "Moneylender",
                  "Bard", "Conclave", "Cursed Village",
                  "Devil's Workshop", "Tragic Hero"],
    },

    # ── Dominion & Renaissance ───────────────────────────────────────
    {
        "name": "It Takes a Villager",
        "expansions": ["Dominion", "Renaissance"],
        "cards": ["Market", "Merchant", "Mine", "Smithy", "Vassal",
                  "Acting Troupe", "Cargo Ship", "Recruiter", "Seer",
                  "Treasurer"],
    },
    {
        "name": "Capture the Flag",
        "expansions": ["Dominion", "Renaissance"],
        "cards": ["Cellar", "Festival", "Harbinger", "Remodel", "Workshop",
                  "Flag Bearer", "Lackeys", "Scholar", "Swashbuckler",
                  "Villain"],
    },

    # ── Dominion & Menagerie ─────────────────────────────────────────
    {
        "name": "Pony Express",
        "expansions": ["Dominion", "Menagerie"],
        "cards": ["Artisan", "Cellar", "Market", "Mine", "Village",
                  "Barge", "Destrier", "Paddock", "Stockpile", "Supplies"],
    },
    {
        "name": "Garden of Cats",
        "expansions": ["Dominion", "Menagerie"],
        "cards": ["Bandit", "Gardens", "Harbinger", "Merchant", "Moat",
                  "Black Cat", "Displace", "Sanctuary", "Scrap",
                  "Snowy Village"],
    },

    # ── Empires & Dark Ages ──────────────────────────────────────────
    {
        "name": "Tomb of the Rat King",
        "expansions": ["Empires", "Dark Ages"],
        "cards": ["Castles", "Chariot Race", "City Quarter", "Legionary",
                  "Sacrifice", "Death Cart", "Fortress", "Pillage",
                  "Rats", "Storeroom"],
    },
    {
        "name": "Triumph of the Bandit King",
        "expansions": ["Empires", "Dark Ages"],
        "cards": ["Capital", "Charm", "Engineer", "Groundskeeper",
                  "Legionary", "Bandit Camp", "Catacombs",
                  "Hunting Grounds", "Market Square", "Procession"],
    },
    {
        "name": "The Squire's Ritual",
        "expansions": ["Empires", "Dark Ages"],
        "cards": ["Archive", "Catapult", "Crown", "Patrician",
                  "Settlers", "Feodum", "Hermit", "Ironmonger",
                  "Rogue", "Squire"],
    },

    # ── Empires & Adventures ─────────────────────────────────────────
    {
        "name": "Area Control",
        "expansions": ["Empires", "Adventures"],
        "cards": ["Capital", "Catapult", "Charm", "Crown",
                  "Farmers' Market", "Coin of the Realm", "Page",
                  "Relic", "Treasure Trove", "Wine Merchant"],
    },
    {
        "name": "No Money No Problems",
        "expansions": ["Empires", "Adventures"],
        "cards": ["Archive", "Encampment", "Royal Blacksmith",
                  "Temple", "Villa", "Dungeon", "Duplicate",
                  "Hireling", "Peasant", "Transmogrify"],
    },
]
