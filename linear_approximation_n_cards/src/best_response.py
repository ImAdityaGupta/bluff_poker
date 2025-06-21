from setup import Strategy, StrategyAllStorage, sampler


def best_response_builder(opp_strat, env_facts, training_params, logging_tools):
    n, k = env_facts["n"], env_facts["k"]
    br_strategy = StrategyAllStorage(n, k)










# def sarsa_one(n, k, episode_len, test_snapshot_interval, brain_snapshot_interval, epsilon_control, alpha_control):
#     snapshots_q_list = []
#     snapshots_n_list = []
#     # brain_snapshot_episodes = []
#     # test_snapshot_episodes = []
#     decision_snapshots = []
#     loss = []
#
#     for episode in range(episode_len):
#         if episode % brain_snapshot_interval == 0:
#             # Make a copy of the entire 3D arrays
#             snapshots_q_list.append(q_space.copy())
#             snapshots_n_list.append(n_space.copy())
#             # brain_snapshot_episodes.append(episode)
#             decision_snapshots.append(get_decision_probs(n, k, q_space))
#
#         if episode % test_snapshot_interval == 0:
#             test = evaluator(n, k)
#             # test = deep_eval(n, k, q_space)
#             this_loss = -test
#             loss.append(this_loss)
#             # test_snapshot_episodes.append(episode)
#
#         net_cards, strat_cards = deal_init_cards(n)
#
#         epsilon = get_epsilon(epsilon_control, episode, episode_len)
#
#         states, actions, reward_ = sampler(n=n, k=k, net_cards=net_cards, strat_cards=strat_cards, epsilon=epsilon)
#
#         for t in range(len(actions)):
#             state = states[t]
#             cards = state[0]
#             immediate_history = state[1]
#             action = actions[t]
#             reward = reward_ if t == len(actions) - 1 else 0
#
#             n_space[tuple(cards)][immediate_history][action] += 1
#             visited_num = n_space[tuple(cards)][immediate_history][action]
#
#             alpha = get_alpha(alpha_control, visited_num, episode, episode_len)
#
#             next_state = states[t + 1]
#             strat_next_move = next_state[1]
#
#             would_be_next_action = greedy_choice(n, k, net_cards, strat_next_move)
#
#             # q_space[tuple(cards)][immediate_history][action] = q_space[tuple(cards)][immediate_history][action] + (
#             #         (reward - q_space[tuple(cards)][immediate_history][action]) * 1 / visited_num)
#             q_space[tuple(cards)][immediate_history][action] = q_space[tuple(cards)][immediate_history][action] + (
#                     (reward - q_space[tuple(cards)][immediate_history][action] +
#                      q_space[tuple(cards)][strat_next_move][would_be_next_action]) * alpha)
#
#     snapshots_q = np.stack(snapshots_q_list, axis=0)  # shape=(num_snapshots, dim1, dim2, dim3)
#     snapshots_n = np.stack(snapshots_n_list, axis=0)  # same shape
#     return snapshots_q, snapshots_n, decision_snapshots, loss
#     # return snapshots_q, snapshots_n, test_snapshot_episodes, brain_snapshot_episodes, decision_snapshot_episodes, loss
#




