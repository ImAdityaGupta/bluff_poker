import os
import numpy as np
from setup import StrategyAllStorage_MC, HumanDebugInputStrategy, sampler

import os
import pickle

def one_iter_logging(opp_strat, exp_num, iter_num, logs, env_facts, num_iters, eta_control, rl_training_params):
    """
    Saves all exploitation inputs to pickle files located in:
        experiments/exp_{exp_num}/iter_{iter_num}/run_{run}.pkl

    Each pickle contains a dict with keys:
      - test_snapshots
      - weights_snapshots
      - prob_vector_snapshots
      - eta_control
      - rl_training_params
      - env_facts
      - num_iters
    """
    base_folder = "experiments"
    exp_folder  = os.path.join(base_folder, f"exp_{exp_num}")
    iter_folder = os.path.join(exp_folder, f"iter_{iter_num}")
    os.makedirs(iter_folder, exist_ok=True)

    num_runs = len(logs[0])
    for run in range(num_runs):
        data = {
            'opp_strat':             opp_strat,
            'test_snapshots':        logs[0][run],
            'weights_snapshots':     logs[1][run],
            'prob_vector_snapshots': logs[2][run],
            'eta_control':           eta_control,
            'rl_training_params':    rl_training_params,
            'env_facts':             env_facts,
            'num_iters':             num_iters,
        }
        file_path = os.path.join(iter_folder, f"run_{run}.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_path = os.path.join(iter_folder, f"opp_strat.pkl")
    data = {
        'weight_matrices': opp_strat.weight_matrices,
        'prob_vector': opp_strat.prob_vector,
        'env_facts': env_facts,
    }
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_one_run(exp_num, iter_num, run_num):
    """
    Reads back the pickle file created by `one_iter_logging_pickle`.

    Returns the dict with keys:
      - 'test_snapshots'
      - 'weights_snapshots'
      - 'prob_vector_snapshots'
      - 'eta_control'
      - 'rl_training_params'
      - 'env_facts'
      - 'num_iters'
    """
    base_folder = "experiments"
    exp_folder = os.path.join(base_folder, f"exp_{exp_num}")
    iter_folder = os.path.join(exp_folder, f"iter_{iter_num}")

    if run_num == -1:
        file_path = os.path.join(iter_folder, "opp_strat.pkl")
    else:
        file_path = os.path.join(iter_folder, f"run_{run_num}.pkl")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    return data


def strat_from_weights(exp_num, iter_num, run_num):
    data = read_one_run(exp_num, iter_num, run_num)
    n, k = data['env_facts']['n'], data['env_facts']['k']

    loaded_strat = StrategyAllStorage_MC(n, k)

    if run_num == -1:
        loaded_strat.weight_matrices = data["weight_matrices"]
        loaded_strat.current_weights = loaded_strat.weight_matrices[-1]
        loaded_strat.prob_vector = data["prob_vector"]


    else:

        loaded_strat.weight_matrices = data["weights_snapshots"][-1]
        loaded_strat.current_weights = loaded_strat.weight_matrices[-1]
        loaded_strat.prob_vector = data["prob_vector_snapshots"][-1]

    return loaded_strat







if __name__ == "__main__":
    loaded_strat = strat_from_weights(0,24,-1)
    while True:
        print(sampler(3, 2, loaded_strat, HumanDebugInputStrategy(3,2), debug=True))

