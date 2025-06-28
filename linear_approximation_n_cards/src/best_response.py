import copy
import random
import numpy as np
import math
import pandas as pd
from setup import StrategyAllStorage_MC, RandomStrategy, HumanDebugInputStrategy, TellTruthSimpleStrategy, sampler, eval_strats, copy_array_to_clipboard
from copy import deepcopy
from logging_weights import strat_from_weights

def get_alpha(alpha_control, episode, episode_len):
    # return 0.1
    # hyperparameters (tweak to taste)
    alpha0     = 0.2        # starting LR
    decay_rate = 0.5        # LR ← LR * decay_rate at each step
    # decay 3 times over the full run:
    step_size  = max(1, episode_len // 3)

    # how many decays have we applied so far?
    num_steps = episode // step_size

    return alpha0 * (decay_rate ** num_steps)


def best_response_builder(opp_strat, env_facts, training_params, logging_params):
    """
    Given fixed opp_strat, uses StrategyAllStorage_MC.reinforce_update to generate a 'maximal' exploiter.
    Uses training_params as given.
    Returns Strategy object (in fact StrategyAllStorage_MC object), logs
    """


    n, k = env_facts["n"], env_facts["k"]


    keys = ["num_runs", "episode_len", "method", "epsilon_control", "alpha_control"]
    num_runs, episode_len, method, epsilon_control, alpha_control = map(training_params.get, keys)

    keys = ["test_snapshot_interval", "brain_snapshot_interval", "exp_num"]
    test_snapshot_interval, brain_snapshot_interval, exp_num = map(logging_params.get, keys)

    test_snapshots = []
    weights_snapshots = []
    prob_vector_snapshots = []

    best_br_strategy = None
    best_br_strategy_score = 0


    for run in range(num_runs):
        br_strategy = StrategyAllStorage_MC(n, k)
        this_run_test_snapshots = []
        this_run_weights_snapshots = []
        this_run_prob_vector_snapshots = []

        for episode in range(episode_len):
            if episode % test_snapshot_interval == 0:
                test_result = eval_strats(n, k, br_strategy, opp_strat)
                this_run_test_snapshots.append((episode, test_result[3]))
                print(test_result[3])

            if episode % brain_snapshot_interval == 0:
                this_run_weights_snapshots.append(copy.deepcopy(br_strategy.weight_matrices))
                this_run_prob_vector_snapshots.append(copy.deepcopy(br_strategy.prob_vector))

            trajectories = []
            br_goes_first = True if random.randint(0, 1) else False
            who_goes_first = 0 if br_goes_first else 1

            if br_goes_first:
                history, [player_one_cards, player_two_cards], winner = sampler(n, k, br_strategy, opp_strat)
            else:
                history, [player_one_cards, player_two_cards], winner = sampler(n, k, opp_strat, br_strategy)

            br_cards = player_one_cards if br_goes_first else player_two_cards
            start = 0 if br_goes_first else 1
            for idx in range(start, len(history), 2):
                history_without_last_action = history[: idx]
                taken_action = history[idx]
                _, probs, mask, features = br_strategy.sample_action(history_without_last_action, br_cards, who_goes_first)

                trajectories.append((history_without_last_action, br_cards, who_goes_first, taken_action, features, probs))

            reward = 1.0 if winner == who_goes_first else -1.0
            alpha = get_alpha(alpha_control, episode, episode_len)
            br_strategy.reinforce_update(trajectories, reward, alpha)

        test_result = eval_strats(n, k, br_strategy, opp_strat, repeats=20)
        this_run_test_snapshots.append((episode_len, test_result[3]))
        this_run_weights_snapshots.append(copy.deepcopy(br_strategy.weight_matrices))
        this_run_prob_vector_snapshots.append(copy.deepcopy(br_strategy.prob_vector))
        if test_result[3] > best_br_strategy_score:
            best_br_strategy = br_strategy
            best_br_strategy_score = test_result[3]

        print(f"this run: score = {test_result}\n")

        test_snapshots.append(this_run_test_snapshots)
        weights_snapshots.append(this_run_weights_snapshots)
        prob_vector_snapshots.append(this_run_prob_vector_snapshots)

    return best_br_strategy, (test_snapshots, weights_snapshots, prob_vector_snapshots)








if __name__ == "__main__":

    temp_n = 3
    temp_k = 2

    env_facts = {"n": temp_n, "k": temp_k}
    training_params = {
        "num_runs": 10,
        "episode_len": 60_000,
        "method": None,
        "epsilon_control": None,
        "alpha_control": None,
    }
    logging_params = {"test_snapshot_interval": 1000, "brain_snapshot_interval": 1000, "exp_num": 0}


    current_strat = strat_from_weights(1, 99, -1)

    max_exploiter, (test_snapshots_, brain_snapshots_, prob_vector_snapshots_) = best_response_builder(current_strat, env_facts, training_params, logging_params)






