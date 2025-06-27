import random
import numpy as np
import math
import pandas as pd

from setup import StrategyAllStorage_MC, RandomStrategy, HumanDebugInputStrategy, TellTruthSimpleStrategy, sampler, eval_strats, copy_array_to_clipboard

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



def best_response_builder(opp_strat, env_facts, training_params, logging_tools):
    n, k = env_facts["n"], env_facts["k"]
    br_strategy = StrategyAllStorage_MC(n, k)

    keys = ["num_runs", "episode_len", "method", "epsilon_control", "alpha_control"]
    num_runs, episode_len, method, epsilon_control, alpha_control = map(training_params.get, keys)

    keys = ["test_snapshot_interval", "brain_snapshot_interval", "cycle_name", "cycle_iter"]
    test_snapshot_interval, brain_snapshot_interval, cycle_name, cycle_iter = map(logging_params.get, keys)

    for episode in range(episode_len):
        if episode % test_snapshot_interval == 0:
            print(eval_strats(n, k, br_strategy, opp_strat))

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

    return br_strategy

if __name__ == "__main__":

    temp_n = 3
    temp_k = 2

    env_facts = {"n": temp_n, "k": temp_k}
    training_params = {
        "num_runs": 1,
        "episode_len": 20_000,
        "method": None,
        "epsilon_control": None,
        "alpha_control": None,
    }
    logging_params = {"test_snapshot_interval": 1000, "brain_snapshot_interval": 1000, "cycle_name": "cyc_000",
                      "cycle_iter": "iter_000"}


    current_strat = best_response_builder(RandomStrategy(temp_n, temp_k), env_facts, training_params, logging_params)
    copy_array_to_clipboard(current_strat.current_weights)
    print("")

    for x in range(10):
        best_exploiter = best_response_builder(current_strat, env_facts, training_params, logging_params)
        current_strat.weight_matrices.append(best_exploiter.current_weights)
        current_strat.current_weights = current_strat.weight_matrices[-1]

        # eta = 1/(2+x)
        #
        # current_strat.prob_vector = [(1-eta)*p for p in current_strat.prob_vector]
        # current_strat.prob_vector.append(1-sum(current_strat.prob_vector))

        iter_num = len(current_strat.weight_matrices)
        current_strat.prob_vector = [1/iter_num for i in range(iter_num)]


        copy_array_to_clipboard(current_strat.current_weights)
        print("")

    while True:
        print(sampler(temp_n, temp_k, current_strat, HumanDebugInputStrategy(temp_n, temp_k)))

        print(sampler(temp_n, temp_k, HumanDebugInputStrategy(temp_n, temp_k), current_strat))


    # pot_best = best_response_builder(strat1, env_facts, training_params, logging_params)




    # copy_array_to_clipboard(pot_best.current_weights)





