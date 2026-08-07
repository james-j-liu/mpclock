"""TrueSkill rating engine + pair selection, mirroring FedLock.

Ratings live on a 0-100 mu scale: start mu=50, sigma=8.333. Each pairwise
"which is more hawkish" outcome is a 1v1 match where the hawkish winner beats
the other. TrueSkill updates both ratings; sigma shrinks as evidence accrues.

Pair selection blends three strategies (fractions from config):
  - Swiss-style: pair speeches with nearby mu (informative, close matches)
  - Uncertainty-targeted: prioritise speeches with the highest sigma
  - Random: keep the comparison graph connected / unbiased
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import trueskill


@dataclass
class Rating:
    mu: float
    sigma: float


class Tournament:
    def __init__(self, ids: list[str], *, initial_mu=50.0, initial_sigma=8.333,
                 beta=None, tau=None, seed=0):
        # beta scales skill->win prob; tau adds per-game dynamics. Keep TS defaults
        # proportional to our wider (0-100) scale.
        beta = beta if beta is not None else initial_sigma / 2
        tau = tau if tau is not None else initial_sigma / 100
        self.env = trueskill.TrueSkill(
            mu=initial_mu, sigma=initial_sigma, beta=beta, tau=tau, draw_probability=0.0
        )
        self.ratings: dict[str, trueskill.Rating] = {
            i: self.env.create_rating() for i in ids
        }
        self.n_comp: dict[str, int] = {i: 0 for i in ids}
        self.rng = random.Random(seed)

    def record(self, winner_id: str, loser_id: str) -> None:
        """winner = the more hawkish speech."""
        w, l = self.ratings[winner_id], self.ratings[loser_id]
        new_w, new_l = self.env.rate_1vs1(w, l)
        self.ratings[winner_id], self.ratings[loser_id] = new_w, new_l
        self.n_comp[winner_id] += 1
        self.n_comp[loser_id] += 1

    def rating(self, i: str) -> Rating:
        r = self.ratings[i]
        return Rating(mu=r.mu, sigma=r.sigma)

    def max_sigma(self) -> float:
        return max(r.sigma for r in self.ratings.values())

    def mean_appearances(self) -> float:
        return sum(self.n_comp.values()) / max(1, len(self.n_comp))

    # ---- pair selection ----
    def select_pairs(self, n_pairs: int, fractions: dict) -> list[tuple[str, str]]:
        ids = list(self.ratings)
        pairs: list[tuple[str, str]] = []

        n_unc = int(n_pairs * fractions.get("uncertainty_fraction", 0.3))
        n_swiss = int(n_pairs * fractions.get("swiss_fraction", 0.6))
        n_rand = n_pairs - n_unc - n_swiss

        # uncertainty-targeted: high-sigma speeches matched to a near-mu partner
        by_sigma = sorted(ids, key=lambda i: -self.ratings[i].sigma)
        by_mu = sorted(ids, key=lambda i: self.ratings[i].mu)
        mu_index = {i: k for k, i in enumerate(by_mu)}
        for i in by_sigma[: n_unc * 2]:
            if len(pairs) >= n_unc:
                break
            j = self._near_mu_partner(i, by_mu, mu_index)
            if j:
                pairs.append((i, j))

        # swiss-style: shuffle, sort by mu, pair neighbours with small jitter
        order = sorted(ids, key=lambda i: self.ratings[i].mu)
        k = 0
        while len([p for p in pairs]) < n_unc + n_swiss and k + 1 < len(order):
            a, b = order[k], order[k + 1]
            if a != b:
                pairs.append((a, b))
            k += 2
            if k + 1 >= len(order):
                self.rng.shuffle(order)
                k = 0

        # random
        for _ in range(max(0, n_rand)):
            a, b = self.rng.sample(ids, 2)
            pairs.append((a, b))

        self.rng.shuffle(pairs)
        return pairs[:n_pairs]

    def _near_mu_partner(self, i, by_mu, mu_index):
        idx = mu_index[i]
        for delta in (1, -1, 2, -2, 3, -3):
            j_idx = idx + delta
            if 0 <= j_idx < len(by_mu) and by_mu[j_idx] != i:
                return by_mu[j_idx]
        return None
