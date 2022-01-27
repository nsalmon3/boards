from turtle import clone
from tensorflow.keras import Model
from tensorflow.keras.layers import (Input, Dense, Conv2D, Flatten, Concatenate)
from tensorflow.keras.models import clone_model
import json
import random

from connect_4.implementations import *
from tree import mcts_node, mcts_training_player

def naive_decision_function(root_board: connect_4_board, _board: connect_4_board):
    if _board.id == BLACK_WINS:
        if root_board.id == BLACK_TO_MOVE or root_board.id == RED_OFFERS_DRAW or root_board.id == BLACK_WINS:
            v = 1
        elif root_board.id == RED_TO_MOVE or root_board.id == BLACK_OFFERS_DRAW or root_board.id == RED_WINS:
            v = -1
        else:
            raise ValueError("This is only reached if DRAW happens downstream from BLACK_WINS. This should not be possible!")
        p = []
    elif _board.id == RED_WINS:
        if root_board.id == BLACK_TO_MOVE or root_board.id == RED_OFFERS_DRAW or root_board.id == BLACK_WINS:
            v = -1
        elif root_board.id == RED_TO_MOVE or root_board.id == BLACK_OFFERS_DRAW or root_board.id == RED_WINS:
            v = 1
        else:
            raise ValueError("This is only reached if DRAW happens downstream from RED_WINS. This should not be possible!")
        p = []
    else:
        v = 0
        _moves = _board.moves
        if OFFER_DRAW in _moves: _moves.remove(OFFER_DRAW)
        if DECLINE_DRAW in _moves: _moves.remove(DECLINE_DRAW)
        if RESIGN in _moves: _moves.remove(RESIGN)
        _len = len(_moves)
        p = list(mcts_node(_move, P = 1 / _len) for _move in _moves)

    return {
        "value": v,
        "policy": p
    }

# The number of games for a training network to play against itself before retraining and evaluation
_SELF_PLAY_GAMES = 100

# The number of games that will be written to disk for future training sessions
_GAMES_STORED = 10000

# The number of boards to be sampled from the _GAMES_STORED games for weight training
_BATCH_SIZE = 1000

# The number of games to play against the best network to evaluate a newly trained network
_EVALUTATION_GAMES = 100

# The percentage of games that must be won by a new network to be declared the new best network
_WIN_THRESHOLD = 0.55

# The relative directory to store games
_GAMES_DIR = "connect_4\\games\\"

class connect_4_model(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Inputs
        self.board_input = Input(shape = (6, 7), name = "board")
        self.id_input = Input(shape = (7, ), name = 'id')

        # Layers for board convolution embedding
        self.convolutional_1 = Conv2D(64, (3, 3), padding = "same")
        self.convolutional_2 = Conv2D(32, (3, 3), padding = "same")

        # Layers after concatenation
        self.dense_1 = Dense(16)
        self.dense_2 = Dense(16)
        self.dense_3 = Dense(16)
        self.policy_head = Dense(7 + 4, activation = 'softmax')
        self.value_head = Dense(1, activation='tanh')
    
    def call(self, inputs):
        board_embedding = self.convolutional_1(inputs["board"])
        board_embedding = self.convolutional_2(board_embedding)
        board_embedding = Flatten()(board_embedding)

        x = Concatenate()([board_embedding, inputs["id"]])
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        return {
            "value": self.value_head(x),
            "policy": self.policy_head(x)
        }

class mcts_trainer():
    def __init__(self,
        _model: connect_4_model,
        game_pool: str = "best") -> None:
        """
        game_pool: This is a name for the pool of games the trainer will use for training.
            If no game_pool.json file exists in _GAMES_DIR, then a new one will be made.
        """

        # Start by initializing the model
        self.model = _model
        self.model.build({
            "board": (None, 6, 7, 1),
            "id": (None, 7)
        })

        # Now try to load in games in the given game_pool, defaulting to a fresh pool if the file doesn't exist.
        try:
            with open(_GAMES_DIR + game_pool, "r") as f:
                self.game_pool = json.load(f)
        except FileNotFoundError:
            self.game_pool = {
                "games": []
            }
        
        # Deserialize the boards
        for _game in self.game_pool['games']:
            for _board in _game['boards']:
                _board["board"] = connect_4_board.deserialize(_board["ser"])

    @property
    def decision_func(self):
        def f(root_board: connect_4_board, _board: connect_4_board):
            # Call the model on the virtual board
            r = self.model({
                "board": _board.grid.reshape((1, 6, 7, 1)).astype(float),
                "id": ID_ONE_HOT[_board.id].reshape((1, 7)).astype(float)
            })
            value_head = r["value"][0].numpy()[0]
            policy_head = r["policy"][0].numpy()

            # Process the value for the current board state
            # The first if means it is black's turn, so the network outputs their true value
            # If it's not black's turn, then negate the value so it's from the persepctive of red
            # Technically the draw one is ambiguous...
            if root_board.id == BLACK_TO_MOVE or root_board.id == RED_OFFERS_DRAW or root_board.id == RED_WINS or root_board.id == DRAW:
                v = value_head
            else:
                v = -value_head
            
            # Process the policy
            p = list()
            for _move in _board.moves:
                p.append(mcts_node(
                    _move = _move,
                    P = np.dot(policy_head, MOVE_ONE_HOT[_move])
                ))
            
            return {
                "value": v,
                "policy": p
            }
        return f
    
    def training_round(self) -> dict:
        """
        The main training loop of the entire mcts algorithm.
        """

        # Initialize
        _board = connect_4_board()
        trainee = mcts_training_player(_board, self.decision_func)
        trainee_copy = mcts_training_player(_board, self.decision_func)
        _game = connect_4_game(_board, trainee, trainee_copy)

        # We beginning training with self play.
        for __game in _game.mcts_play(_SELF_PLAY_GAMES)["games"]:
            self.game_pool["games"].append(__game)
        
        # Now prep the training data
        games_batch = random.sample(self.game_pool, _BATCH_SIZE)

        x_train = []
        y_train = []
        for _game in games_batch:
            _x_board = random.sample(_game["boards"], 1)
            x_train.append({
                "board": _x_board.grid.reshape((6, 7, 1)).astype(float),
                "id": ID_ONE_HOT[_x_board.id].reshape((7, 1)).astype(float)
            })
            y_train.append({
                "value": _game["result"],
                "policy": _x_board.policy
            })
        
        training_model = clone_model(self.model)

        # Now actually train the model
        training_model.fit(x = x_train, y = y_train)

        # Now evaluate the model against the best model
        _game = connect_4_game(_board, trainee, mcts_player(_board, self.decision_func))
        draws = _game.play_tourney(_EVALUTATION_GAMES)

        if trainee.wins / _EVALUTATION_GAMES > _WIN_THRESHOLD:
            self.model = trainee

        return {
            "fuckin A man": "fuck"
        }