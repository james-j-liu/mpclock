"""Monetary Policy Committee membership roster (single source of truth).

The corpus is scraped from every speech on the Bank of England website, which
publishes talks by many officials who never vote on Bank Rate — prudential
regulation (PRA), financial stability (FPC), markets and banking operations,
banknotes, and so on. Only the Monetary Policy Committee sets Bank Rate, so only
its members (and the "BoE MPC" composite) belong in the scoring system.

The MPC has nine members: the Governor, the Deputy Governors (Monetary Policy,
Financial Stability, and — since 2014 — Markets & Banking), the Chief Economist,
and four external members appointed by the Chancellor. This module is the
authoritative membership list; the pipeline filters the scoring pool through
`is_mpc`, and the site builds its Rankings "Members" filter from `to_dict()`
(emitted into data.json meta.roster), so the two never drift.

CURRENT = MPC members in office as of 2026. FORMER = people who served on the
MPC at any point over 1997-2026 but have since left.
"""
from __future__ import annotations

BOE_MPC = "BoE MPC"

CURRENT_MPC = {
    "Andrew Bailey",        # Governor
    "Clare Lombardelli",    # Deputy Governor, Monetary Policy
    "Sarah Breeden",        # Deputy Governor, Financial Stability
    "Dave Ramsden",         # Deputy Governor, Markets & Banking
    "Huw Pill",             # Chief Economist
    "Catherine Mann", "Swati Dhingra", "Megan Greene", "Alan Taylor",  # external members
}

FORMER_MPC = {
    # Governors
    "Edward George", "Mervyn King", "Mark Carney",
    # Deputy Governors (Monetary Policy / Financial Stability / Markets & Banking)
    "Howard Davies", "David Clementi", "Andrew Large", "Rachel Lomax", "John Gieve",
    "Paul Tucker", "Charles Bean", "Charlie Bean", "Ben Broadbent", "Jon Cunliffe",
    "Nemat Shafik",
    # Chief Economists / internal executive members
    "John Vickers", "Charles Goodhart", "Ian Plenderleith", "Willem Buiter",
    "DeAnne Julius", "Charles Bean", "Spencer Dale", "Andrew Haldane",
    # external members
    "Alan Budd", "Sushil Wadhwani", "Christopher Allsopp", "Stephen Nickell",
    "Kate Barker", "Marian Bell", "Richard Lambert", "David Walton",
    "David Blanchflower", "Tim Besley", "Andrew Sentance", "Adam Posen",
    "Paul Fisher", "Martin Weale", "David Miles", "Ian McCafferty",
    "Kristin Forbes", "Gertjan Vlieghe", "Michael Saunders", "Silvana Tenreyro",
    "Jonathan Haskel",
}

# Spelling variants in the corpus that refer to one canonical person (honorifics,
# middle initials, and outright misspellings the scraper picks up from titles).
ALIASES = {
    "Eddie George": "Edward George",
    "Sir Edward George": "Edward George",
    "Andy Haldane": "Andrew Haldane",
    "Charlie Bean": "Charles Bean",
    "Minouche Shafik": "Nemat Shafik",
    "Catherine L. Mann": "Catherine Mann",
    "Catherine L Mann": "Catherine Mann",
    "Sir Jon Cunliffe": "Jon Cunliffe",
    "Sir Dave Ramsden": "Dave Ramsden",
    "Timothy Besley": "Tim Besley",
    "Andrew Sentence": "Andrew Sentance",
    "Adam S. Posen": "Adam Posen",
    "David G Blanchflower": "David Blanchflower",
    "Professor Stephen Nickell": "Stephen Nickell",
    "Willem H. Buiter": "Willem Buiter",
    "Sir Alan Budd": "Alan Budd",
    "Mark Carney At": "Mark Carney",
    # BIS byline spellings (middle initials, honorifics)
    "Andrew G Haldane": "Andrew Haldane",
    "Andrew G. Haldane": "Andrew Haldane",
    "David Ramsden": "Dave Ramsden",
    "Sir Andrew Large": "Andrew Large",
    "Sir David Clementi": "David Clementi",
    "Sir Howard Davies": "Howard Davies",
    "Sir John Vickers": "John Vickers",
    "Sir Paul Tucker": "Paul Tucker",
    "Sir Charles Bean": "Charles Bean",
    "Sir Mervyn King": "Mervyn King",
    "Sir Andrew Bailey": "Andrew Bailey",
}


def canon(name: str) -> str:
    return ALIASES.get(name, name)


def is_current_mpc(name: str) -> bool:
    n = canon(name)
    return n == BOE_MPC or n in CURRENT_MPC


def is_mpc(name: str) -> bool:
    """True for any MPC member (current or former) and the MPC composite."""
    n = canon(name)
    return n == BOE_MPC or n in CURRENT_MPC or n in FORMER_MPC


def to_dict() -> dict:
    """Roster payload for the site (embedded in data.json meta)."""
    return {
        "current": sorted(CURRENT_MPC),
        "former": sorted(FORMER_MPC),
        "aliases": ALIASES,
    }
