"""Bidding agents competing over repeated auctions.

- TruthfulAgent bids its true value per click (dominant in SP/VCG).
- ShadingAgent learns a first-price shading factor online: it shades more after
  winning with margin (payment < shaded bid) and shades less after losing. Truthful
  bidding is NOT an equilibrium in first-price auctions, so surplus-maximizing
  agents must shade.
- BudgetPacedAgent throttles its bid by 1/(1 + lambda) and adapts lambda with a
  dual-variable update on spend error, so total spend lands near its daily budget.
- RandomAgent is a control with uniform random bids around the value.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from mua.auction.types import Advertiser, Allocation, Bid, Candidate


class BiddingAgent(ABC):
    agent_type: str = "base"

    def __init__(self, advertiser: Advertiser, seed: int = 0):
        self.advertiser = advertiser
        self.rng = np.random.default_rng(seed)
        self.spend = 0.0
        self.clicks = 0
        self.conversions = 0.0
        self.surplus = 0.0
        self.wins = 0
        self.rounds = 0

    @abstractmethod
    def make_bid(self, candidate: Candidate) -> Bid: ...

    def observe(self, allocation: Allocation | None, click: bool) -> None:
        self.rounds += 1
        if allocation is not None:
            self.wins += 1
            self.spend += allocation.payment * allocation.pctr
            self.surplus += (self.advertiser.value_per_click - allocation.payment) * allocation.pctr
            if click:
                self.clicks += 1
                self.conversions += self.advertiser.conversion_rate

    def tick(self, round_idx: int, total_rounds: int) -> None:  # noqa: ARG002
        """End-of-round hook; used by pacing agents."""
        return None


class TruthfulAgent(BiddingAgent):
    agent_type = "truthful"

    def make_bid(self, candidate: Candidate) -> Bid:
        return Bid(
            advertiser_id=self.advertiser.advertiser_id,
            merchant_id=candidate.merchant_id,
            value=self.advertiser.value_per_click,
            amount=self.advertiser.value_per_click,
            pctr=candidate.pctr,
            organic_relevance=candidate.organic_relevance,
        )


class RandomAgent(BiddingAgent):
    agent_type = "random"

    def __init__(self, advertiser: Advertiser, seed: int = 0, low: float = 0.5, high: float = 1.5):
        super().__init__(advertiser, seed)
        self.low = low
        self.high = high

    def make_bid(self, candidate: Candidate) -> Bid:
        return Bid(
            advertiser_id=self.advertiser.advertiser_id,
            merchant_id=candidate.merchant_id,
            value=self.advertiser.value_per_click,
            amount=float(self.advertiser.value_per_click * self.rng.uniform(self.low, self.high)),
            pctr=candidate.pctr,
            organic_relevance=candidate.organic_relevance,
        )


class ShadingAgent(BiddingAgent):
    """Multiplicative-weights shading for first-price auctions.

    Wins with positive margin mean the shaded bid was above the clearing price, so
    shading can be increased; losses mean the bid was too shaded.
    """

    agent_type = "shading"

    def __init__(
        self,
        advertiser: Advertiser,
        seed: int = 0,
        initial: float = 0.8,
        step: float = 0.02,
        s_min: float = 0.3,
        s_max: float = 1.0,
    ):
        super().__init__(advertiser, seed)
        self.shading = initial
        self.step = step
        self.s_min = s_min
        self.s_max = s_max

    def make_bid(self, candidate: Candidate) -> Bid:
        return Bid(
            advertiser_id=self.advertiser.advertiser_id,
            merchant_id=candidate.merchant_id,
            value=self.advertiser.value_per_click,
            amount=float(self.shading * self.advertiser.value_per_click),
            pctr=candidate.pctr,
            organic_relevance=candidate.organic_relevance,
        )

    def observe(self, allocation: Allocation | None, click: bool) -> None:
        super().observe(allocation, click)
        if allocation is None:
            self.shading = max(self.s_min, self.shading * (1 - self.step))
        else:
            margin = self.shading * self.advertiser.value_per_click - allocation.payment
            if margin > 1e-12:
                self.shading = min(self.s_max, self.shading * (1 + self.step))


class BudgetPacedAgent(BiddingAgent):
    """Paces spend toward the daily budget with a dual-variable lambda.

    Bids value / (1 + lambda). lambda updates each round on the relative error
    between the observed spend rate and the target rate implied by the remaining
    budget, so cumulative spend converges to (at most) the budget.
    """

    agent_type = "paced"

    def __init__(
        self,
        advertiser: Advertiser,
        seed: int = 0,
        learning_rate: float = 0.3,
        lam_init: float = 0.0,
        lam_max: float = 10.0,
    ):
        super().__init__(advertiser, seed)
        self.lam = lam_init
        self.learning_rate = learning_rate
        self.lam_max = lam_max
        self._prev_spend = 0.0

    def make_bid(self, candidate: Candidate) -> Bid:
        return Bid(
            advertiser_id=self.advertiser.advertiser_id,
            merchant_id=candidate.merchant_id,
            value=self.advertiser.value_per_click,
            amount=float(self.advertiser.value_per_click / (1.0 + self.lam)),
            pctr=candidate.pctr,
            organic_relevance=candidate.organic_relevance,
        )

    def tick(self, round_idx: int, total_rounds: int) -> None:
        spend_rate = self.spend - self._prev_spend
        self._prev_spend = self.spend
        remaining = self.advertiser.daily_budget - self.spend
        remaining_rounds = max(total_rounds - round_idx - 1, 1)
        target_rate = remaining / remaining_rounds
        if target_rate > 0:
            error = (spend_rate - target_rate) / target_rate
            self.lam = float(min(self.lam_max, max(0.0, self.lam + self.learning_rate * error)))
