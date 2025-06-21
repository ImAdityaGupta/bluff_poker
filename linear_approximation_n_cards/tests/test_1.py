# tests/test_game_state.py
import pytest
import random

from linear_approximation_n_cards.src.setup import GameState

@pytest.fixture(autouse=True)
def seed_rand():
    # keep tests deterministic
    random.seed(42)
    yield
    random.seed()

def test_deal_correct_counts():
    gs = GameState(n=5, k=3)
    gs.init_distribute_cards()
    assert len(gs.player_one_cards) == 3, "Player one should get k cards"
    assert len(gs.player_two_cards) == 3, "Player two should get k cards"

def test_no_overlap_between_players():
    gs = GameState(n=5, k=4)
    gs.init_distribute_cards()
    p1 = set(gs.player_one_cards)
    p2 = set(gs.player_two_cards)
    assert p1.isdisjoint(p2), "Players should not share any picked positions"

def test_positions_within_range():
    n, k = 7, 5
    gs = GameState(n=n, k=k)
    gs.init_distribute_cards()
    total = 4 * n
    for pos in gs.player_one_cards + gs.player_two_cards:
        assert 0 <= pos < total, f"Position {pos} out of valid range [0, {total})"

def test_reproducible_with_same_seed():
    # Seed, deal once
    random.seed(42)
    gs1 = GameState(n=6, k=4)
    gs1.init_distribute_cards()

    # Reseed with the same seed, deal again
    random.seed(42)
    gs2 = GameState(n=6, k=4)
    gs2.init_distribute_cards()

    # Now they should match exactly
    assert gs1.player_one_cards == gs2.player_one_cards
    assert gs1.player_two_cards == gs2.player_two_cards


