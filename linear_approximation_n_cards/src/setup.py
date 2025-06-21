from abc import ABC, abstractmethod
import random

class Strategy(ABC):
    def __init__(self, n, k):
        self.n = n
        self.k = k

    @abstractmethod
    def next_move(self, player_state):
        pass

class StrategyAllStorage(Strategy):
    def __init__(self, n, k):
        super().__init__(n, k)
        self.prob_vector = []
        self.weights_vector = []
        self.initialize_weights()
        self.initialize_probs()

    def initialize_weights(self):
        pass

    def initialize_probs(self):
        pass

    def next_move(self, player_state):
        # Example placeholder logic
        return "some move"

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

def sampler(n, k, strat1, strat2, check_legal=False):
    '''
    We always assume strat1 goes first.
    returns history, [player_one_cards, player_two_cards], winner (0 or 1, strat1 or strat2)
    '''
    game_state = GameState(n, k)
    player_one_state = PlayerState(n, k)
    player_two_state = PlayerState(n, k)

    player_one_state.player_cards = game_state.player_one_cards
    player_two_state.player_cards = game_state.player_two_cards

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
                print(game_state.game_state_readable(winner))
                return game_state.history, [game_state.player_one_cards, game_state.player_two_cards], winner

        current_player = 1 - current_player
        # print(game_state.history)

strat1 = TellTruthSimpleStrategy(3, 2)
strat2 = HumanDebugInputStrategy(3, 2)


print(sampler(3, 2, strat1, strat2))

