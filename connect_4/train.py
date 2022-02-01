from tensorflow.keras import Model
from tensorflow.keras.layers import (Input, Dense, Conv2D, Flatten, Concatenate)
from tensorflow.keras.losses import (Loss, CategoricalCrossentropy, MeanSquaredError)
from tensorflow.keras.models import clone_model
import json
import random

from connect_4.implementations import *
from tree import (mcts_node, mcts)

def naive_decision_function(root_board: connect_4_board, _board: connect_4_board):
    if _board.bid == BLACK_WINS:
        if root_board.bid == BLACK_TO_MOVE or root_board.bid == RED_OFFERS_DRAW or root_board.bid == BLACK_WINS:
            v = 1
        elif root_board.bid == RED_TO_MOVE or root_board.bid == BLACK_OFFERS_DRAW or root_board.bid == RED_WINS:
            v = -1
        else:
            raise ValueError("This is only reached if DRAW happens downstream from BLACK_WINS. This should not be possible!")
        p = []
    elif _board.bid == RED_WINS:
        if root_board.bid == BLACK_TO_MOVE or root_board.bid == RED_OFFERS_DRAW or root_board.bid == BLACK_WINS:
            v = -1
        elif root_board.bid == RED_TO_MOVE or root_board.bid == BLACK_OFFERS_DRAW or root_board.bid == RED_WINS:
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
_SAMPLE_SIZE = 1000

# The number of games to play against the best network to evaluate a newly trained network
_EVALUTATION_GAMES = 100

# The percentage of games that must be won by a new network to be declared the new best network
_WIN_THRESHOLD = 0.55

# The relative directory to store games
_GAMES_DIR = "connect_4\\games\\"

_mean_squared_error = MeanSquaredError()
_categorical_cross_entropy = CategoricalCrossentropy()

class _default_loss(Loss):
    def call(self, y_true, y_pred):
        return _mean_squared_error.call(y_true['value'], y_pred['value']) + _categorical_cross_entropy(y_true['policy'], y_pred['policy'])

class connect_4_model(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Inputs
        self.grid_input = Input(shape = (6, 7), name = "grid")
        self.bid_input = Input(shape = (7, ), name = 'bid')

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
        board_embedding = self.convolutional_1(inputs["grid"])
        board_embedding = self.convolutional_2(board_embedding)
        board_embedding = Flatten()(board_embedding)

        x = Concatenate()([board_embedding, inputs["bid"]])
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        return {
            "value": self.value_head(x),
            "policy": self.policy_head(x)
        }
    
    def copy(self):
        new_model = type(self)()
        new_model.build({
            'grid': (None, 6, 7, 1),
            'bid': (None, 7)
        })
        new_model.compile(loss = _default_loss)
        new_model.set_weights(self.get_weights())
        return new_model
    
    def board_call(self, _board: connect_4_board):
        grid = _board.grid.reshape((1,6,7,1))
        _id = _board.bid.as_array.reshape((1,7))

        r = self.call({'grid': grid, 'bid': _id})
        _policy = r['policy']

        return {
            "value": r["value"][0,0].numpy(),
            "policy": {
                PLACE_1: _policy[0, PLACE_1.index].numpy(),
                PLACE_2: _policy[0, PLACE_2.index].numpy(),
                PLACE_3: _policy[0, PLACE_3.index].numpy(),
                PLACE_4: _policy[0, PLACE_4.index].numpy(),
                PLACE_5: _policy[0, PLACE_5.index].numpy(),
                PLACE_6: _policy[0, PLACE_6.index].numpy(),
                PLACE_7: _policy[0, PLACE_7.index].numpy(),

                OFFER_DRAW: _policy[0, OFFER_DRAW.index].numpy(),
                ACCEPT_DRAW: _policy[0, ACCEPT_DRAW.index].numpy(),
                DECLINE_DRAW: _policy[0, DECLINE_DRAW.index].numpy(),
                RESIGN: _policy[0, RESIGN.index].numpy()
            }
        }

class mcts_trainer():
    def __init__(self, _model: connect_4_model, game_pool_name: str = "best") -> None:
        """
        game_pool: This is a name for the pool of games the trainer will use for training.
            If no game_pool.json file exists in _GAMES_DIR, then a new one will be made.
        """

        # Start by initializing the model
        self.model = _model
        self.model.compile(loss = _default_loss)

        # Now try to load in games in the given game_pool, defaulting to a fresh pool if the file doesn't exist.
        self.game_pool_name = game_pool_name
        try:
            with open(_GAMES_DIR + game_pool_name, "r") as f:
                self.game_pool = json.load(f)
        except FileNotFoundError:
            self.game_pool = {
                "games": []
            }
        
        # Deserialize the boards
        for _game in self.game_pool['games']:
            for _board in _game['boards']:
                _board["board"] = connect_4_board.deserialize(_board["serialized"])

    @property
    def decision_func(self):
        def f(root_board: connect_4_board, _board: connect_4_board):
            # Call the model on the virtual board
            r = self.model.board_call(_board)
            value_head = r["value"]
            policy_head = r["policy"]

            # Process the value for the current board state
            # The first if means it is black's turn, so the network outputs their true value
            # If it's not black's turn, then negate the value so it's from the persepctive of red
            # Technically the draw one is ambiguous...
            if root_board.bid == BLACK_TO_MOVE or root_board.bid == RED_OFFERS_DRAW or root_board.bid == RED_WINS or root_board.bid == DRAW:
                v = value_head
            else:
                v = -value_head
            
            # Process the policy
            p = list()
            for _move in _board.moves:
                p.append(mcts_node(
                    _move = _move,
                    P = policy_head[_move]
                ))
            
            return {
                "value": v,
                "policy": p
            }
        return f

    @staticmethod
    def gen_decision_func(model: connect_4_model):
        def f(root_board: connect_4_board, _board: connect_4_board):
            # Call the model on the virtual board
            r = model.board_call(_board)
            value_head = r["value"]
            policy_head = r["policy"]

            # Process the value for the current board state
            # The first if means it is black's turn, so the network outputs their true value
            # If it's not black's turn, then negate the value so it's from the persepctive of red
            # Technically the draw one is ambiguous...
            if root_board.bid == BLACK_TO_MOVE or root_board.bid == RED_OFFERS_DRAW or root_board.bid == RED_WINS or root_board.bid == DRAW:
                v = value_head
            else:
                v = -value_head
            
            # Process the policy
            p = list()
            for _move in _board.moves:
                p.append(mcts_node(
                    _move = _move,
                    P = np.dot(policy_head, _move.as_array)
                ))
            
            return {
                "value": v,
                "policy": p
            }
        return f
    
    def save_games(self):
        serialized = list()
        for game in self.game_pool['games']:
            serialized.append({
                'b_and_p': {
                    'board': game['board'].serialize(),
                    'policy': game['policy'].tolist()
                },
                'value': game['value'].tolist()
            })
        with open(_GAMES_DIR + self.game_pool_name, 'w') as f:
            f.truncate(0)
            json.dump({
                'games': serialized
            }, f)

    def self_play(self, rounds: int = _SELF_PLAY_GAMES):
        for _ in range(rounds):
            # We being by initializing two instances of mcts on a shared board
            b = connect_4_board()
            m1 = mcts(b, self.decision_func)
            m2 = mcts(b, self.decision_func)

            # We keep track of the boards in this game here.
            game = {
                'b_and_p': list(),
                'value': None
            }

            # This iterable will be our looping mechanism
            it = iter(cycle([m1, m2]))
            current_mcts = next(it)
            while not b.is_terminal:
                # Get a move from the current mcts
                _dict = current_mcts.run_stochastically()

                _move = _dict['move']
                all_nodes = _dict['all_nodes'] # This is a list of all the nodes that were possible moves, with true probabilities attached.

                # Now that the mcts has finished, we want to add the true probabilities to training data
                p = np.zeros((7 + 4,), dtype = float) # start by setting all probabilities equal to zero
                for _node in all_nodes:
                    p[_node.move.index] = _node.P # recall that each move class has an attr index for this exact moment!

                # Now append the board and it's true probabilities
                game['b_and_p'].append({
                    'board': b.copy(),
                    'policy': p
                })

                # Now make the move on the board
                b.move(_move)

                # Move to the next tree
                current_mcts = next(it)

                # Inform it of the previous move that was made
                current_mcts.move(_move)
            
            # Once the game is over, we want to record the result
            # We store it in a numpy array because the neural network prefers this
            # When we serialize things will be stored differently
            if b.bid == BLACK_WINS:
                game['value'] = np.array([1.0])
            elif b.bid == RED_WINS:
                game['value'] = np.array([-1.0])
            else:
                game['value'] = np.array([0.0])
            
            # Finally add this game to the game pool
            self.game_pool['games'].append(game)

        # Once all the games have been played, prune the game pool to only include the desired history
        self.game_pool['games'] = self.game_pool['games'][-_GAMES_STORED:]
    
    def train_and_eval(self,
        sample_size: int = _SAMPLE_SIZE,
        batch_size: int = 256,
        epochs: int = 64,
        eval_games: int = _EVALUTATION_GAMES,
        thresh: float = _WIN_THRESHOLD):

        # In the train and eval step, we copy the best model, train it, and play the copy against the best
        trainee = self.model.copy()

        # Pick our random games and set up the training data
        x_train = {
            'grid': np.empty((sample_size, 6, 7, 1)),
            'bid': np.empty((sample_size, 7))
        }
        y_train = {
            'value': np.empty((sample_size,)),
            'policy': np.empty((sample_size, 7 + 4))
        }
        training_games = np.random.choice(self.game_pool['games'], sample_size)
        for i, game in enumerate(training_games):
            _dict = np.random.choice(game['b_and_p'])
            x_train['grid'][i] = _dict['board'].grid.reshape((6, 7, 1))
            x_train['bid'][i] = _dict['board'].bid.as_array
            y_train['value'][i] = game['value']
            y_train['policy'][i] = _dict['policy']
        
        # Actually train the model on this data
        trainee.fit(x = x_train, y = y_train, batch_size = batch_size, epochs = epochs)
        
        # Now play the trainee against the best model
        b = connect_4_board()
        trainee_player = mcts_player(b, self.gen_decision_func(trainee))
        best_player = mcts_player(b, self.decision_func)

        trainee_player.wins = 0
        best_player.wins = 0
        draws = 0

        for n in range(eval_games):
            if n % 2 == 0:
                black_player = trainee_player
                red_player = best_player
            else:
                black_player = best_player
                red_player = trainee_player

            # This iterable will be our looping mechanism
            it = iter(cycle([black_player, red_player]))
            current_player = next(it)
            while not b.is_terminal:
                # Get a move from the current player
                _move = current_player.move()

                # Now make the move on the board
                b.move(_move)

                # Move to the next player
                current_player= next(it)

                # Inform it of the previous move that was made
                current_player.inform(_move)
            
            # Once the game is over, we want to record the result
            # We store it in a numpy array because the neural network prefers this
            # When we serialize things will be stored differently
            if b.bid == BLACK_WINS:
                black_player.wins += 1
            elif b.bid == RED_WINS:
                red_player.wins += 1
            else:
                draws += 1
        
        # Now we check to see if the trainee player won enough games and declare it best if so
        if trainee_player.wins / eval_games > thresh:
            self.model = trainee