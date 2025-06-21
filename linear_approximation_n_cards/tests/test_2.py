import pytest
import random

# Adjust the import path to your module as needed
from linear_approximation_n_cards.src.setup import (
    RandomStrategy,
    TellTruthSimpleStrategy,
    PlayerState,
    call_index,
    sampler,
    GameState,
)

@pytest.fixture(autouse=True)
def seed_rng():
    random.seed(42)
    yield
    random.seed()


def test_random_strategy_first_move_range():
    n, k = 4, 2
    strat = RandomStrategy(n, k)
    ps = PlayerState(n, k)
    ps.history = []
    # first move should be in [0 .. 3n-1]
    move = strat.next_move(ps)
    assert 0 <= move < call_index(n, k), "First move out of bounds"


def test_random_strategy_subsequent_move_range():
    n, k = 4, 2
    strat = RandomStrategy(n, k)
    ps = PlayerState(n, k)
    # simulate first move
    ps.history = [5]
    move = strat.next_move(ps)
    assert 5 < move <= call_index(n, k), "Subsequent move illegal"


def test_tell_truth_simple_initial_move():
    n, k = 5, 3
    strat = TellTruthSimpleStrategy(n, k)
    ps = PlayerState(n, k)
    # give a hand with two 2s and one 4
    ps.player_cards = [2, 7, 14]  # mod n => [2, 2, 4]
    ps.history = []
    # should pick rank=2, count=2 => claim = 2 + 5*(2-1) = 7
    move = strat.next_move(ps)
    assert move == 7


def test_tell_truth_simple_with_higher_last_move():
    n, k = 5, 3
    strat = TellTruthSimpleStrategy(n, k)
    ps = PlayerState(n, k)
    # hand has three 1s
    ps.player_cards = [1, 6, 11]
    # beliefs => rank 1 count=3 => capped => 3 => code = 1 + 5*(3-1) = 11
    ps.history = [5]  # last move = 5
    move = strat.next_move(ps)
    assert move == 11, "Should pick the highest confident claim"


def test_tell_truth_simple_calls_when_no_improvement():
    n, k = 5, 3
    strat = TellTruthSimpleStrategy(n, k)
    ps = PlayerState(n, k)
    # hand has one of each rank: max_value=1 => choose the highest rank (4)
    ps.player_cards = [0, 6, 12]  # mod 5 => [0,1,2]
    # last move is above our biggest_claim=2 + 5*0 = 2 -> e.g. 3
    ps.history = [3]
    move = strat.next_move(ps)
    assert move == call_index(n, k), "Should call when no higher claim is valid"


def test_sampler_basic_structure_and_winner():
    n, k = 3, 2
    strat1 = RandomStrategy(n, k)
    strat2 = RandomStrategy(n, k)
    # with seed and two randoms, we expect a valid game end
    history, cards, winner = sampler(n, k, strat1, strat2)
    # history ends with a call
    assert history[-1] == call_index(n, k)
    # each hand size is k
    assert len(cards[0]) == k
    assert len(cards[1]) == k
    # winner is 0 or 1
    assert winner in (0, 1)


