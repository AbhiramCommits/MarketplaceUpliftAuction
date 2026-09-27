"""Game-theory checks that the auction mechanisms are implemented correctly.

- Second-price and VCG: unilateral deviation from truthful bidding must never
  increase expected surplus (verified across a grid of deviations and many
  random value/pCTR scenarios).
- First-price: truthful bidding must be beatable by shading (truthful surplus is
  always zero because the winner pays its own bid; shading leaves positive surplus).
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from mua.auction.agents import BudgetPacedAgent, TruthfulAgent
from mua.auction.mechanisms import MECHANISMS, VCG, FirstPrice, SecondPrice
from mua.auction.types import Advertiser, AuctionRequest, Bid, Candidate

DEVIATIONS = (0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.8, 2.5)


def _surplus(
    mechanism, candidates, bids, advertiser_id: str, value: float, slots: tuple[int, ...]
) -> float:
    request = AuctionRequest(
        request_id=0,
        candidates=tuple(candidates),
        reserve=0.0,
        slot_positions=slots,
    )
    outcome = mechanism.allocate(request, bids)
    for allocation in outcome.allocations:
        if allocation.advertiser_id == advertiser_id:
            return (value - allocation.payment) * allocation.pctr
    return 0.0


def _make_bids(values, pctrs) -> list[Bid]:
    return [
        Bid(
            advertiser_id=f"a{i}",
            merchant_id=i,
            value=values[i],
            amount=values[i],
            pctr=pctrs[i],
            organic_relevance=pctrs[i],
        )
        for i in range(len(values))
    ]


class TestTruthfulDominance:
    def test_second_price_truthful_is_dominant(self):
        rng = np.random.default_rng(42)
        mechanism = SecondPrice()
        for scenario in range(40):
            values = rng.uniform(0.5, 4.0, 3)
            pctrs = rng.uniform(0.01, 0.2, 3)
            candidates = [
                Candidate(merchant_id=i, pctr=pctrs[i], organic_relevance=pctrs[i])
                for i in range(3)
            ]
            baseline = _make_bids(values, pctrs)
            for i in range(3):
                truthful = _surplus(mechanism, candidates, baseline, f"a{i}", values[i], (1,))
                for factor in DEVIATIONS:
                    deviated = list(baseline)
                    deviated[i] = Bid(
                        advertiser_id=f"a{i}",
                        merchant_id=i,
                        value=values[i],
                        amount=values[i] * factor,
                        pctr=pctrs[i],
                        organic_relevance=pctrs[i],
                    )
                    surplus = _surplus(mechanism, candidates, deviated, f"a{i}", values[i], (1,))
                    assert surplus <= truthful + 1e-9, (
                        scenario,
                        i,
                        factor,
                        surplus,
                        truthful,
                    )

    def test_vcg_truthful_is_dominant(self):
        rng = np.random.default_rng(7)
        for n_bidders, slots in ((4, (1, 2)), (5, (1, 2, 3))):
            mechanism = VCG()
            for scenario in range(30):
                values = rng.uniform(0.5, 4.0, n_bidders)
                pctrs = rng.uniform(0.01, 0.2, n_bidders)
                candidates = [
                    Candidate(merchant_id=i, pctr=pctrs[i], organic_relevance=pctrs[i])
                    for i in range(n_bidders)
                ]
                baseline = _make_bids(values, pctrs)
                for i in range(n_bidders):
                    truthful = _surplus(mechanism, candidates, baseline, f"a{i}", values[i], slots)
                    for factor in DEVIATIONS:
                        deviated = list(baseline)
                        deviated[i] = Bid(
                            advertiser_id=f"a{i}",
                            merchant_id=i,
                            value=values[i],
                            amount=values[i] * factor,
                            pctr=pctrs[i],
                            organic_relevance=pctrs[i],
                        )
                        surplus = _surplus(
                            mechanism, candidates, deviated, f"a{i}", values[i], slots
                        )
                        assert surplus <= truthful + 1e-9, (
                            n_bidders,
                            slots,
                            scenario,
                            i,
                            factor,
                        )


class TestFirstPriceShading:
    def test_truthful_is_beatable_by_shading(self):
        rng = np.random.default_rng(0)
        mechanism = FirstPrice()
        shaded_surpluses = []
        truthful_surpluses = []
        for _scenario in range(40):
            v1, v2 = rng.uniform(1.0, 4.0), rng.uniform(0.5, 3.0)
            p1 = p2 = 0.05
            candidates = [
                Candidate(merchant_id=0, pctr=p1, organic_relevance=p1),
                Candidate(merchant_id=1, pctr=p2, organic_relevance=p2),
            ]
            truthful_bids = _make_bids([v1, v2], [p1, p2])
            truthful = _surplus(mechanism, candidates, truthful_bids, "a0", v1, (1,))
            truthful_surpluses.append(truthful)

            shaded_bids = list(truthful_bids)
            shaded_bids[0] = Bid(
                advertiser_id="a0",
                merchant_id=0,
                value=v1,
                amount=0.7 * v1,
                pctr=p1,
                organic_relevance=p1,
            )
            shaded = _surplus(mechanism, candidates, shaded_bids, "a0", v1, (1,))
            shaded_surpluses.append(shaded)

        assert all(t == 0.0 for t in truthful_surpluses)
        assert np.mean(shaded_surpluses) > 0.0
        assert sum(s > 0 for s in shaded_surpluses) >= 1
        assert np.mean(shaded_surpluses) > np.mean(truthful_surpluses)

    def test_mutual_shading_still_beats_truthful(self):
        mechanism = FirstPrice()
        pctr = 0.1
        candidates = [
            Candidate(merchant_id=0, pctr=pctr, organic_relevance=pctr),
            Candidate(merchant_id=1, pctr=pctr, organic_relevance=pctr),
        ]
        bids = _make_bids([2.0, 1.5], [pctr, pctr])
        both_shaded = list(bids)
        both_shaded[0] = Bid("a0", 0, 2.0, 1.4, pctr, pctr)
        both_shaded[1] = Bid("a1", 1, 1.5, 1.05, pctr, pctr)
        winner_surplus = _surplus(mechanism, candidates, both_shaded, "a0", 2.0, (1,))
        assert winner_surplus == (2.0 - 1.4) * pctr
        assert winner_surplus > 0.0


class TestMechanismProperties:
    """Property-based checks of the payment rules (hypothesis)."""

    @st.composite
    def auction_inputs(draw):
        n = draw(st.integers(1, 8))
        amounts = draw(
            st.lists(
                st.floats(0.1, 5.0, allow_nan=False, allow_infinity=False), min_size=n, max_size=n
            )
        )
        pctrs = draw(
            st.lists(
                st.floats(0.005, 0.2, allow_nan=False, allow_infinity=False), min_size=n, max_size=n
            )
        )
        reserve = draw(st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False))
        mechanism_name = draw(st.sampled_from(["first_price", "second_price", "gsp", "vcg"]))
        if mechanism_name == "second_price":
            slots = (1,)
        else:
            slots = draw(st.sampled_from([(1,), (1, 2), (1, 2, 3)]))
        return mechanism_name, amounts, pctrs, reserve, slots

    def _allocate(self, mechanism_name, amounts, pctrs, reserve, slots):
        bids = [
            Bid(
                advertiser_id=f"a{i}",
                merchant_id=i,
                value=amounts[i],
                amount=amounts[i],
                pctr=pctrs[i],
                organic_relevance=pctrs[i],
            )
            for i in range(len(amounts))
        ]
        request = AuctionRequest(
            request_id=0,
            candidates=tuple(
                Candidate(merchant_id=b.merchant_id, pctr=b.pctr, organic_relevance=b.pctr)
                for b in bids
            ),
            reserve=reserve,
            slot_positions=slots,
        )
        outcome = MECHANISMS[mechanism_name]().allocate(request, bids)
        return bids, request, outcome

    @given(auction_inputs())
    @settings(max_examples=200, deadline=None)
    def test_payment_never_exceeds_bid(self, inp):
        mechanism_name, amounts, pctrs, reserve, slots = inp
        bids, _, outcome = self._allocate(mechanism_name, amounts, pctrs, reserve, slots)
        bid_by_id = {b.advertiser_id: b.amount for b in bids}
        for allocation in outcome.allocations:
            assert allocation.payment <= bid_by_id[allocation.advertiser_id] + 1e-9

    @given(auction_inputs())
    @settings(max_examples=200, deadline=None)
    def test_allocation_is_valid_permutation(self, inp):
        mechanism_name, amounts, pctrs, reserve, slots = inp
        bids, request, outcome = self._allocate(mechanism_name, amounts, pctrs, reserve, slots)
        advertisers = [a.advertiser_id for a in outcome.allocations]
        positions = [a.slot for a in outcome.allocations]
        assert len(set(advertisers)) == len(advertisers)
        assert len(set(positions)) == len(positions)
        assert set(positions) <= set(request.slot_positions)
        eligible = sorted(
            (b for b in bids if b.amount >= reserve - 1e-12 and b.pctr > 0),
            key=lambda b: (-b.amount * b.pctr, b.advertiser_id),
        )
        expected = [b.advertiser_id for b in eligible[: len(slots)]]
        assert advertisers == expected
        for allocation in outcome.allocations:
            assert allocation.payment >= reserve - 1e-9

    @given(auction_inputs())
    @settings(max_examples=200, deadline=None)
    def test_gsp_never_charges_more_than_own_bid(self, inp):
        mechanism_name, amounts, pctrs, reserve, slots = inp
        if mechanism_name != "gsp":
            return
        bids, _, outcome = self._allocate(mechanism_name, amounts, pctrs, reserve, slots)
        bid_by_id = {b.advertiser_id: b.amount for b in bids}
        for allocation in outcome.allocations:
            assert allocation.payment <= bid_by_id[allocation.advertiser_id] + 1e-9


class TestBudgetPacing:
    def test_paced_agent_spends_within_5_percent_of_budget(self):
        value, pctr, opponent_value = 3.0, 0.1, 1.0
        rounds = 3000
        # If the paced agent won every round it would pay the opponent's eCPM (1.0)
        # per click; the budget targets 70% of that.
        budget = 0.7 * opponent_value * pctr * rounds
        advertiser = Advertiser(
            advertiser_id="paced",
            merchant_id=1,
            value_per_click=value,
            daily_budget=budget,
            conversion_rate=0.08,
        )
        paced = BudgetPacedAgent(advertiser, seed=0, learning_rate=0.3)
        opponent = Advertiser(
            advertiser_id="truthful",
            merchant_id=2,
            value_per_click=opponent_value,
            daily_budget=1e9,
            conversion_rate=0.08,
        )
        truthful = TruthfulAgent(opponent, seed=1)
        mechanism = SecondPrice()
        candidate_p = Candidate(merchant_id=1, pctr=pctr, organic_relevance=pctr)
        candidate_o = Candidate(merchant_id=2, pctr=pctr, organic_relevance=pctr)
        candidates = (candidate_p, candidate_o)

        for t in range(rounds):
            bid_p = paced.make_bid(candidate_p)
            bid_o = truthful.make_bid(candidate_o)
            request = AuctionRequest(
                request_id=t, candidates=candidates, reserve=0.0, slot_positions=(1,)
            )
            outcome = mechanism.allocate(request, [bid_p, bid_o])
            paced_allocation = next(
                (a for a in outcome.allocations if a.advertiser_id == "paced"), None
            )
            paced.observe(paced_allocation, False)
            paced.tick(t, rounds)

        assert abs(paced.spend - budget) / budget <= 0.05, (
            paced.spend,
            budget,
            paced.lam,
        )
