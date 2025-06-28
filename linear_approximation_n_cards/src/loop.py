from setup import StrategyAllStorage_MC, RandomStrategy, HumanDebugInputStrategy, TellTruthSimpleStrategy, sampler, eval_strats, copy_array_to_clipboard
from setup import save_all_storage_mc_strategy_npz, load_all_storage_mc_strategy_npz
from best_response import best_response_builder, get_alpha
from logging_weights import one_iter_logging

def get_eta(eta_control, iter, num_iters):
    return min(0.5, 4/(1+iter))


def experiment(env_facts, num_iters, eta_control, rl_training_params, rl_logging_params, initial_strat=None, logging=True):
    n, k = env_facts["n"], env_facts["k"]


    keys = ["num_runs", "episode_len", "method", "epsilon_control", "alpha_control"]
    num_runs, episode_len, method, epsilon_control, alpha_control = map(rl_training_params.get, keys)

    keys = ["test_snapshot_interval", "brain_snapshot_interval", "exp_num"]
    test_snapshot_interval, brain_snapshot_interval, exp_num = map(rl_logging_params.get, keys)

    if initial_strat is None:
        current_strat = StrategyAllStorage_MC(n, k)
    else:
        current_strat = initial_strat



    for iter in range(num_iters):
        best_exploiter, logs = best_response_builder(current_strat, env_facts, rl_training_params, rl_logging_params)
        if logging:
            one_iter_logging(current_strat, exp_num, iter, logs, env_facts, num_iters, eta_control, rl_training_params)

        current_strat.merge_weights(best_exploiter, get_eta(eta_control, iter, num_iters))







    return current_strat









if __name__ == "__main__":

    temp_n = 10
    temp_k = 2

    env_facts = {"n": temp_n, "k": temp_k}
    training_params = {
        "num_runs": 1,
        "episode_len": 10_000,
        "method": None,
        "epsilon_control": None,
        "alpha_control": None,
    }
    logging_params = {"test_snapshot_interval": 1000, "brain_snapshot_interval": 1000, "exp_num": 2}


    final_strat = experiment(env_facts, 10, None, training_params, logging_params, logging=True)

    # save_all_storage_mc_strategy_npz("my_strat.npz", final_strat.n, final_strat.k, final_strat.prob_vector, final_strat.weight_matrices)
    #
    # loaded = load_all_storage_mc_strategy_npz("my_strat.npz")

    print("")





    # while True:
    #     print(sampler(temp_n, temp_k, current_strat, HumanDebugInputStrategy(temp_n, temp_k), debug=True))
    #
    #     print(sampler(temp_n, temp_k, HumanDebugInputStrategy(temp_n, temp_k), current_strat, debug=True))





