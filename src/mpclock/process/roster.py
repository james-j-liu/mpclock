"""Roster of Bank of England / UK officials, used to drive anonymization.

The roster is the union of:
  - speaker names observed in the built corpus, and
  - a curated seed list of well-known BoE Governors, Deputy Governors, Chief
    Economists and MPC members whose names recur inside other people's speeches.

We keep this list deliberately broad: the anonymizer needs *every* name that a
judge might recognise, not just the speech's own author. Place/concept names that
collide with surnames (Phillips curve, Taylor rule) are protected by EXCLUSIONS.
"""
from __future__ import annotations

# Curated seed of names likely to appear inside speeches/Q&A. Extend freely.
SEED_OFFICIALS: list[str] = [
    # Governors
    "Edward George", "Eddie George", "Mervyn King", "Mark Carney", "Andrew Bailey",
    # Deputy Governors (Monetary Policy / Financial Stability / Markets & Banking)
    "David Clementi", "Rachel Lomax", "John Gieve", "Charlie Bean", "Charles Bean",
    "Paul Tucker", "Jon Cunliffe", "Ben Broadbent", "Dave Ramsden", "Sam Woods",
    "Sarah Breeden", "Clare Lombardelli", "Nemat Shafik", "Minouche Shafik",
    # Chief Economists
    "John Vickers", "Charles Goodhart", "Spencer Dale", "Andrew Haldane", "Andy Haldane",
    "Huw Pill",
    # External MPC members (selected, historical + recent)
    "Willem Buiter", "Sushil Wadhwani", "DeAnne Julius", "Christopher Allsopp",
    "Stephen Nickell", "Kate Barker", "Marian Bell", "Richard Lambert", "David Walton",
    "David Blanchflower", "Tim Besley", "Andrew Sentance", "Adam Posen", "Paul Fisher",
    "Martin Weale", "Ben Broadbent", "David Miles", "Ian McCafferty", "Kristin Forbes",
    "Gertjan Vlieghe", "Michael Saunders", "Silvana Tenreyro", "Jonathan Haskel",
    "Catherine Mann", "Swati Dhingra", "Megan Greene", "Alan Taylor",
    # other recurring senior officials
    "Ian Plenderleith", "Andrew Large", "Paul Fisher", "Charlie Bean",
]

# Strings that look like surnames but must NOT be redacted (concepts / places).
EXCLUSIONS: set[str] = {
    "phillips", "taylor", "lucas", "fisher", "walton", "bell", "large",
    # UK concept / place names
    "threadneedle", "london", "sterling", "westminster", "mais", "mansion",
}

TITLES = [
    "Governor", "Deputy Governor", "Chief Economist", "Executive Director",
    "President", "Vice-President", "Chair", "Chairman", "Director", "Professor",
    "Sir", "Dame", "Lord", "Baroness", "Mr", "Mr.", "Ms", "Ms.", "Mrs", "Mrs.",
    "Dr", "Dr.",
]


def build_roster(corpus_speakers: list[str]) -> list[str]:
    names = set(SEED_OFFICIALS)
    for s in corpus_speakers:
        if s and s != "BoE MPC":
            names.add(s)
    return sorted(names)
