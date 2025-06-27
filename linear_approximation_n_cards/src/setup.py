from abc import ABC, abstractmethod
import random
import numpy as np
import pandas as pd
import itertools
import math
from collections import Counter
np.set_printoptions(suppress=True, precision=6)

class Strategy(ABC):
    def __init__(self, n, k):
        self.n = n
        self.k = k

    @abstractmethod
    def next_move(self, player_state):
        pass


class StrategyAllStorage_MC(Strategy):
    """
    Monte Carlo strategy with softmax policy parameterized by a weight matrix.

    Attributes:
      n: int, number of distinct card labels
      k: int, hand size
      M: int, number of actions = 3*n + 1
      D: int, feature dimension = 8*n + 4 (includes bias)
      weight_matrices: list of np.ndarray, each shape (M, D)
      current_weights: np.ndarray, shape (M, D)
      prob_vector: placeholder
    """
    def __init__(self, n, k):
        super().__init__(n, k)
        self.M = 3 * n + 1
        self.D = 7 * n + 5
        self.prob_vector = np.array([1.0])
        self.weight_matrices = []
        self.current_weights = None
        self.new_weight_matrix()

        self.which_matrix_index = None

    def next_move(self, player_state):
        history = player_state.history
        player_cards = player_state.player_cards
        who_first = player_state.player_order

        if len(history) == 0 or len(history) == 1:
            self.which_matrix_index = np.random.choice(len(self.weight_matrices), p=self.prob_vector)

        which_matrix = self.weight_matrices[self.which_matrix_index]


        action, probs, _, _ =  self.sample_action(history, player_cards, who_first, which_weights=which_matrix)
        print(probs)
        return action

    def new_weight_matrix(self):
        """
        Initialize a fresh M x D weight matrix with zeros.
        """
        W = np.zeros((self.M, self.D), dtype=float)
        self.weight_matrices.append(W)
        self.current_weights = W

    def player_state_to_feature(self, history, player_cards, who_first):
        """
        Encode game state to features vector of length D = 7*n + 5.
        """
        features = np.zeros(self.D, dtype=float)
        # 1) our and their past moves
        even = history[0::2]
        odd  = history[1::2]
        our_moves   = even if who_first == 0 else odd
        their_moves = odd  if who_first == 0 else even
        for m in our_moves:
            features[m] = 1.0
        for m in their_moves:
            features[m + (3*self.n + 1)] = 1.0
        # 2) hand counts
        offset = 6*self.n + 2
        counts = np.zeros(self.n, int)
        for c in player_cards:
            counts[c % self.n] += 1
        for i in range(self.n):
            if counts[i] >= 1:
                features[offset + i] = 1.0
            if counts[i] >= 2:
                features[offset + self.n] = 1.0
        # 3) who went first
        features[7*self.n + 3] = float(who_first)
        # 4) bias constant
        features[7*self.n + 4] = 1.0
        return features

    def legal_mask(self, history):
        """
        Boolean mask length M: True for legal moves.
        """
        mask = np.zeros(self.M, dtype=bool)
        if not history:
            mask[:self.M-1] = True  # disallow final pass
        else:
            last = history[-1]
            if last + 1 < self.M:
                mask[last+1:] = True
        return mask

    def sample_action(self, history, player_cards, who_first, which_weights=None):
        """
        Sample an action based on current_weights (by default) and softmax.
        Returns (action, probs, mask, features)
        """

        if which_weights is None:
            which_weights = self.current_weights

        features = self.player_state_to_feature(history, player_cards, who_first)
        logits = which_weights.dot(features)
        mask = self.legal_mask(history)
        # mask illegal
        logits = np.where(mask, logits, -np.inf)
        logits -= np.max(logits[mask])
        exp_logits = np.exp(logits) * mask
        probs = exp_logits / np.sum(exp_logits)
        action = np.random.choice(self.M, p=probs)
        return action, probs, mask, features

    def reinforce_update(self, trajectories, reward, alpha):
        """
        Apply REINFORCE update to current_weights.
        trajectories: list of tuples (history, player_cards, who_first, action, features, probs)
        reward: scalar terminal reward
        alpha: learning rate
        """
        for history, player_cards, who_first, action, features, probs in trajectories:
            # gradient w.r.t. logits: grad = one_hot(action) - probs
            grad_logits = -probs.copy()
            grad_logits[action] += 1.0

            # grad_logits += 0.01 * (-np.log(probs + 1e-8) - 1)
            # update weight rows: for each action a: W[a] += alpha * reward * grad_logits[a] * features
            # vectorized as outer product
            grad_W = np.outer(grad_logits, features)
            self.current_weights += alpha * reward * grad_W

    def merge_weights(self, other_strat, eta):
        self.weight_matrices += other_strat.weight_matrices
        self.prob_vector = np.concatenate((self.prob_vector*(1-eta), other_strat.prob_vector*eta))
        self.current_weights = self.weight_matrices[-1]


def save_all_storage_mc_strategy_npz(filepath, n, k, prob_vector, weight_matrices):
    """
    Save strategy state to a .npz archive.

    Parameters
    ----------
    filepath : str
        Path to write (e.g. "my_strat.npz").
    n : int
        Number of labels.
    k : int
        Hand size.
    prob_vector : 1D np.ndarray
        Current probability vector.
    weight_matrices : list of np.ndarray
        Your history of (M × D) matrices.
    """
    savez_kwargs = {
        'n': np.array(n, dtype=int),
        'k': np.array(k, dtype=int),
        'prob_vector': prob_vector,
    }
    for idx, W in enumerate(weight_matrices):
        savez_kwargs[f'W{idx}'] = W
    np.savez(filepath, **savez_kwargs)


def load_all_storage_mc_strategy_npz(filepath):
    """
    Load strategy state from a .npz archive saved by save_strategy_npz.
    Returns StrategyAllStorage_MC object
    """
    data = np.load(filepath)
    n = int(data['n'])
    k = int(data['k'])
    obj = StrategyAllStorage_MC(n, k)
    obj.prob_vector = data['prob_vector']


    # Collect W0, W1, ... in ascending order
    weight_keys = sorted(
        [key for key in data.keys() if key.startswith('W')],
        key=lambda s: int(s[1:])  # extract the index from 'W{idx}'
    )
    obj.weight_matrices = [data[key] for key in weight_keys]
    obj.current_weights = obj.weight_matrices[-1] if obj.weight_matrices else None

    return obj


class RandomStrategy(Strategy):
    def __init__(self, n, k):
        super().__init__(n, k)

    def next_move(self, player_state):
        call_move = call_index(self.n, self.k)

        last_move = -1
        if player_state.history:
            last_move = player_state.history[-1]

        if last_move == -1:
            return random.randint(last_move+1, call_move-1)
        return random.randint(last_move+1, call_move)

class TellTruthSimpleStrategy(Strategy):
    def __init__(self, n, k):
        super().__init__(n, k)

    def next_move(self, player_state):
        beliefs = [0 for i in range(self.n)]
        for card in player_state.player_cards:
            beliefs[card%self.n] += 1
        max_index = -1
        max_value = -1
        for i, card in enumerate(beliefs):
            if card >= max_value:
                max_index = i
                max_value = card

        max_value = min(max_value, 3) ## NO QUADS FOR NOW

        biggest_claim = max_index + self.n*(max_value-1)
        last_move = player_state.history[-1] if player_state.history else -1

        if biggest_claim > last_move:
            return biggest_claim
        return call_index(self.n, self.k)

class HumanDebugInputStrategy(Strategy):
    def __init__(self, n, k):
        super().__init__(n, k)
    def next_move(self, player_state):
        print(player_state.player_state_readable())

        human_input = input("Move?\n")


        return string_to_move(player_state.n, human_input)

        # call_move = call_index(self.n, self.k)
        #
        # last_move = -1
        # if player_state.history:
        #     last_move = player_state.history[-1]
        #
        # if last_move == -1:
        #     return random.randint(last_move+1, call_move-1)
        # return random.randint(last_move+1, call_move)

class GameState():
    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.player_one_cards = []
        self.player_two_cards = []
        self.history = []
        self.init_distribute_cards()

    def init_distribute_cards(self):
        # sanity check
        assert 2*self.k <= 4*self.n, "Not enough cards to deal"

        total = 4 * self.n
        # pick 2k distinct positions in [0..total-1]
        picks = random.sample(range(total), 2*self.k)

        # first k go to player one, next k to player two
        self.player_one_cards = picks[:self.k]
        self.player_two_cards = picks[self.k:]

    def is_over(self):
        if not self.history:
            return False
        if self.history[-1] == call_index(self.n, self.k):
            return True
        return False

    def game_winner(self):
        """
        Determine the winner when a call occurs. Assumes:
        - history is a list of claim codes (ints).
        - players alternate turns, with Player 0 making the first claim.
        - a 'call' claim has code 3n and ends the game.

        Returns:
            0 if Player 0 wins,
            1 if Player 1 wins.

        Raises:
            RuntimeError: if the game is not over or invalid history.
        """
        if not self.is_over():
            raise RuntimeError("Cannot determine winner: the game is not over.")

        # Identify who made the call (history indices 0-based: even => Player 0)
        last_idx = len(self.history) - 1
        caller = 0 if last_idx % 2 == 0 else 1
        opponent = 1 - caller

        # The claim being challenged is the one just before the call
        if last_idx < 1:
            raise RuntimeError("Insufficient history to determine challenged claim.")
        claim_code = self.history[-2]


        if not (0 <= claim_code < call_index(self.n, self.k)):
            raise RuntimeError(f"Invalid claim code {claim_code}")
        rank = claim_code % self.n
        threshold = claim_code // self.n + 1

        # Count actual occurrences of rank across both hands (mod n)
        cards = self.player_one_cards + self.player_two_cards
        count = sum(1 for pos in cards if pos % self.n == rank)

        # If actual count >= threshold, claim was true => caller loses
        if count >= threshold:
            return opponent
        # Otherwise claim was false => caller wins
        return caller


    def validate_move(self, move):
        max_claim = call_index(self.n, self.k)
        last = self.history[-1] if self.history else -1
        if not (last < move <= max_claim):
            raise ValueError(f"Illegal move {move}; must be in ({last}, {max_claim}]")

    def game_state_readable(self, winner=None):

        # game_state.history, [game_state.player_one_cards, game_state.player_two_cards], winner

        to_return = "Player 1 Cards: "
        for card in self.player_one_cards:
            to_return += card_to_string(self.n, card)
            to_return += " "
        to_return += "\n"

        to_return += "Player 2 Cards: "
        for card in self.player_two_cards:
            to_return += card_to_string(self.n, card)
            to_return += " "
        to_return += "\n"

        to_return += "Moves: \n"
        for idx, move in enumerate(self.history):
            if idx % 2 == 0:
                to_return += "0: "
            else:
                to_return += "1: "
            to_return += move_to_string(self.n, move)
            to_return += "\n"

        to_return += f"Winner: {winner}\n"


        return to_return



class PlayerState():
    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.player_cards = []
        # player_order is 0 if Player goes first, 1 if Player goes second
        self.player_order = None
        self.history = []

    def player_state_readable(self):

        # game_state.history, [game_state.player_one_cards, game_state.player_two_cards], winner

        to_return = "Cards: "
        for card in self.player_cards:
            to_return += card_to_string(self.n, card)
            to_return += " "
        to_return += "\n"


        to_return += "Moves: \n"
        for idx, move in enumerate(self.history):
            if idx % 2 == 0:
                to_return += "0: "
            else:
                to_return += "1: "
            to_return += move_to_string(self.n, move)
            to_return += "\n"

        return to_return


def call_index(n, k):
    return 3*n

def card_to_string(n, card):
    return str(card%n)

def move_to_string(n, move):
    if move < n:
        return f"{move%n}H"
    if move < 2*n:
        return f"{move%n}P"
    if move < 3*n:
        return f"{move%n}T"
    return "Call"

def string_to_move(n, move_string):
    bonus = 0
    if move_string == "Call":
        return 3*n
    if move_string[-1] == 'T':
        bonus = 2*n
    elif move_string[-1] == 'P':
        bonus = n

    return bonus + int(move_string[:-1])





def sampler(n, k, strat1, strat2, random_dist=True, forced_player_one=None, forced_player_two=None, check_legal=False, debug=False):
    '''
    We always assume strat1 goes first.
    returns history, [player_one_cards, player_two_cards], winner (0 or 1, strat1 or strat2)
    '''
    game_state = GameState(n, k)
    player_one_state = PlayerState(n, k)
    player_two_state = PlayerState(n, k)

    player_one_state.player_cards = game_state.player_one_cards
    player_two_state.player_cards = game_state.player_two_cards
    if not random_dist:
        player_one_state.player_cards = forced_player_one
        player_two_state.player_cards = forced_player_two
        game_state.player_one_cards = forced_player_one
        game_state.player_two_cards = forced_player_two

    player_one_state.player_order = 0
    player_two_state.player_order = 1

    current_player = 0
    while True:
        if current_player == 0:
            # ASSUME STRATS GIVE LEGAL MOVES
            move = strat1.next_move(player_one_state)
            if check_legal:
                game_state.validate_move(move)

            game_state.history.append(move)
            player_one_state.history.append(move)
            player_two_state.history.append(move)

            if game_state.is_over():
                winner = game_state.game_winner()
                if debug:
                    print(game_state.game_state_readable(winner))
                return game_state.history, [game_state.player_one_cards, game_state.player_two_cards], winner

        elif current_player == 1:
            # ASSUME STRATS GIVE LEGAL MOVES
            move = strat2.next_move(player_two_state)
            if check_legal:
                game_state.validate_move(move)

            game_state.history.append(move)
            player_one_state.history.append(move)
            player_two_state.history.append(move)

            if game_state.is_over():
                winner = game_state.game_winner()
                if debug:
                    print(game_state.game_state_readable(winner))
                return game_state.history, [game_state.player_one_cards, game_state.player_two_cards], winner

        current_player = 1 - current_player
        # print(game_state.history)


def eval_strats(n, k, strat1, strat2, repeats=3):
    """
    Evaluate strat1 vs strat2 by iterating over rank‐only deals,
    weighting each by the number of underlying suit‐assignments.
    Returns weighted wins1, wins2, total_weighted_games, win_rate1.
    """
    M = 4  # suits per rank
    wins1 = wins2 = 0.0
    total_weighted_games = 0.0

    for i in range(repeats):

        # all ways to choose k cards *by rank* for player1 (multisets)
        for p1_ranks in itertools.combinations_with_replacement(range(n), k):
            c1 = Counter(p1_ranks)
            # how many suit‐assignments yield exactly these rank counts?
            ways1 = 1
            for r, cnt in c1.items():
                ways1 *= math.comb(M, cnt)

            # now enumerate p2 rank‐multisets from the remaining suit‐counts
            for p2_ranks in itertools.combinations_with_replacement(range(n), k):
                c2 = Counter(p2_ranks)
                # skip any that overuse a rank’s suits
                valid = True
                ways2 = 1
                for r, cnt2 in c2.items():
                    cnt1 = c1.get(r, 0)
                    if cnt1 + cnt2 > M:
                        valid = False
                        break
                    ways2 *= math.comb(M - cnt1, cnt2)
                if not valid:
                    continue

                # total number of underlying deals for this (p1_ranks,p2_ranks)
                weight = ways1 * ways2

                # --- game 1: strat1 first
                _, _, winner = sampler(
                    n, k, strat1, strat2,
                    random_dist=False,
                    forced_player_one=list(p1_ranks),
                    forced_player_two=list(p2_ranks),
                    check_legal=True
                )
                if winner == 0:
                    wins1 += weight
                else:
                    wins2 += weight
                total_weighted_games += weight

                # --- game 2: strat2 first, swap hands
                _, _, win_sw = sampler(
                    n, k, strat2, strat1,
                    random_dist=False,
                    forced_player_one=list(p2_ranks),
                    forced_player_two=list(p1_ranks),
                    check_legal=True
                )
                # winner_swapped == 0 => strat2; ==1 => strat1
                if win_sw == 1:
                    wins1 += weight
                else:
                    wins2 += weight
                total_weighted_games += weight

    win_rate1 = wins1 / total_weighted_games
    return wins1, wins2, total_weighted_games, win_rate1

def copy_array_to_clipboard(arr: np.ndarray, sep: str = "\t", fmt: str = "{:.8e}", ):
        """
        Copy a 2D numpy array to the system clipboard as sep-delimited text.
        No index or header is included.
        """
        if arr.ndim != 2:
            raise ValueError("Only 2D arrays supported")
        pd.DataFrame(arr).to_clipboard(sep=sep, index=False, header=False)
        print("✓ Array copied to clipboard. Now just Ctrl+V into Excel.")
        for row in arr:
            print(sep.join(fmt.format(x) for x in row))

if __name__ == "__main__":


    temp_n = 3
    temp_k = 2

    strat1 = StrategyAllStorage_MC(temp_n, temp_k)
    strat2 = RandomStrategy(temp_n, temp_k)


    # while True:
    #     print(sampler(temp_n, temp_k, strat1, strat2, debug=True))

    print(eval_strats(temp_n, temp_k, strat1, strat2))
    print(eval_strats(temp_n, temp_k, strat2, strat1))

