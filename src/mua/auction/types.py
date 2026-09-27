"""Dataclasses for the ads auction layer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    """A merchant in the candidate slate with pCTR and organic relevance."""

    merchant_id: int
    pctr: float
    organic_relevance: float


@dataclass(frozen=True)
class Bid:
    """An advertiser's bid on a candidate; amount may differ from true value."""

    advertiser_id: str
    merchant_id: int
    value: float
    amount: float
    pctr: float
    organic_relevance: float


@dataclass(frozen=True)
class Advertiser:
    """Static advertiser parameters (identity + economics)."""

    advertiser_id: str
    merchant_id: int
    value_per_click: float
    daily_budget: float
    conversion_rate: float


@dataclass(frozen=True)
class AuctionRequest:
    """One auction: a slate of candidates plus reserve and sponsored positions."""

    request_id: int
    candidates: tuple[Candidate, ...]
    reserve: float
    slot_positions: tuple[int, ...] = (1, 2, 3)


@dataclass(frozen=True)
class Allocation:
    """Winner assignment: advertiser, slate position, per-click payment."""

    advertiser_id: str
    merchant_id: int
    slot: int
    payment: float
    pctr: float
    bid_amount: float
    e_cpm: float
    organic_relevance: float


@dataclass(frozen=True)
class AuctionOutcome:
    """Per-round result including platform-level metrics."""

    request_id: int
    mechanism: str
    allocations: tuple[Allocation, ...]
    n_candidates: int
    n_bids: int
    k_slots: int
    fill_rate: float
    revenue_expected: float
    organic_displacement: float
