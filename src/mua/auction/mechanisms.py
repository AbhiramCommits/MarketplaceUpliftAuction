"""Auction mechanisms as interchangeable strategy classes.

All mechanisms rank bidders by eCPM = bid x pCTR, enforce a per-click reserve price,
and allocate up to K sponsored slots (positions configurable per request).

Payments (per click):
- FirstPrice: winner pays its own bid.
- SecondPrice (Vickrey, single slot): winner pays the minimum bid needed to keep the
  slot, i.e. second-highest eCPM divided by own pCTR.
- GSP (K slots): slot i pays (bid_{i+1} x pCTR_{i+1}) / pCTR_i, the quality-adjusted
  next-price rule; the last filled slot pays the reserve.
- VCG (K slots): each winner pays the externality it imposes, which in the
  position-free CTR model equals the (K+1)-th eCPM divided by own pCTR.

Game-theory properties are verified empirically in tests/test_incentives.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from mua.auction.types import Allocation, AuctionOutcome, AuctionRequest, Bid


class Mechanism(ABC):
    name: str = "base"
    requires_single_slot: bool = False

    def allocate(self, request: AuctionRequest, bids: Sequence[Bid]) -> AuctionOutcome:
        if self.requires_single_slot and len(request.slot_positions) != 1:
            raise ValueError(f"{self.name} requires exactly one slot position")
        reserve = request.reserve
        eligible = sorted(
            (b for b in bids if b.amount >= reserve - 1e-12 and b.pctr > 0),
            key=lambda b: (-b.amount * b.pctr, b.advertiser_id),
        )
        k = len(request.slot_positions)
        winners = eligible[:k]
        payments = self._payments(winners, eligible, reserve) if winners else []

        allocations = tuple(
            Allocation(
                advertiser_id=bid.advertiser_id,
                merchant_id=bid.merchant_id,
                slot=request.slot_positions[i],
                payment=pay,
                pctr=bid.pctr,
                bid_amount=bid.amount,
                e_cpm=bid.amount * bid.pctr,
                organic_relevance=bid.organic_relevance,
            )
            for i, (bid, pay) in enumerate(zip(winners, payments, strict=True))
        )

        organic = sorted((c.organic_relevance for c in request.candidates), reverse=True)
        displacement = 0.0
        for i, allocation in enumerate(allocations):
            position = request.slot_positions[i]
            r_org = organic[position - 1] if position - 1 < len(organic) else 0.0
            displacement += max(0.0, r_org - allocation.organic_relevance)
        displacement = displacement / len(allocations) if allocations else 0.0

        revenue_expected = sum(a.payment * a.pctr for a in allocations)
        return AuctionOutcome(
            request_id=request.request_id,
            mechanism=self.name,
            allocations=allocations,
            n_candidates=len(request.candidates),
            n_bids=len(bids),
            k_slots=k,
            fill_rate=len(allocations) / k if k else 0.0,
            revenue_expected=revenue_expected,
            organic_displacement=displacement,
        )

    @abstractmethod
    def _payments(self, winners: list[Bid], eligible: list[Bid], reserve: float) -> list[float]: ...


class FirstPrice(Mechanism):
    """Winner pays its own bid."""

    name = "first_price"

    def _payments(self, winners: list[Bid], eligible: list[Bid], reserve: float) -> list[float]:
        return [b.amount for b in winners]


class SecondPrice(Mechanism):
    """Vickrey for a single slot: pay the minimum bid needed to keep the slot."""

    name = "second_price"
    requires_single_slot = True

    def _payments(self, winners: list[Bid], eligible: list[Bid], reserve: float) -> list[float]:
        winner = winners[0]
        if len(eligible) > 1:
            runner_up = eligible[1]
            price = runner_up.amount * runner_up.pctr / winner.pctr
            return [max(reserve, price)]
        return [reserve]


class GSP(Mechanism):
    """Generalized second price: quality-adjusted next-price rule.

    Slot i pays (bid_{i+1} x pCTR_{i+1}) / pCTR_i; the last filled slot prices off
    the next eligible bidder below the slate (if any), floored at the reserve.
    With a single slot this reduces to the Vickrey second price.
    """

    name = "gsp"

    def _payments(self, winners: list[Bid], eligible: list[Bid], reserve: float) -> list[float]:
        payments: list[float] = []
        for i, bid in enumerate(winners):
            nxt = winners[i + 1] if i + 1 < len(winners) else None
            if nxt is None and len(eligible) > len(winners):
                nxt = eligible[len(winners)]
            if nxt is not None:
                payments.append(max(reserve, nxt.amount * nxt.pctr / bid.pctr))
            else:
                payments.append(reserve)
        return payments


class VCG(Mechanism):
    """Vickrey-Clarke-Groves: each winner pays its externality on the others.

    In the position-free CTR model the externality of any winner is the (K+1)-th
    highest eCPM (the bidder who would enter the slate in the winner's absence),
    divided by the winner's own pCTR to convert to a per-click price. When fewer
    than K+1 bidders are eligible there is no displaced bidder, so winners pay
    just the reserve.
    """

    name = "vcg"

    def _payments(self, winners: list[Bid], eligible: list[Bid], reserve: float) -> list[float]:
        if len(eligible) > len(winners):
            displaced = eligible[len(winners)]
            externality = displaced.amount * displaced.pctr
            return [max(reserve, externality / bid.pctr) for bid in winners]
        return [reserve] * len(winners)


MECHANISMS: dict[str, type[FirstPrice] | type[SecondPrice] | type[GSP] | type[VCG]] = {
    "first_price": FirstPrice,
    "second_price": SecondPrice,
    "gsp": GSP,
    "vcg": VCG,
}
