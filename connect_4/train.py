from tensorflow.keras import Model
from tensorflow.keras.activations import (relu)
from tensorflow.keras.layers import (Input, Dense, Conv2D, Flatten, Concatenate, BatchNormalization, Add)
from tensorflow.keras.losses import (CategoricalCrossentropy, MeanSquaredError)
from tensorflow.keras.models import (load_model, clone_model)
from tensorflow.keras.regularizers import (L2)
from abc import *
import json

from connect_4.implementations import *
from tree import (mcts_node, mcts, mcts_player)

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

def naive_mcts_function(root_board: connect_4_board, _board: connect_4_board):
    if _board.bid == BLACK_WINS:
        if root_board.bid == BLACK_TO_MOVE or root_board.bid == BLACK_WINS:
            v = 1
        elif root_board.bid == RED_TO_MOVE or root_board.bid == RED_WINS:
            v = -1
        else:
            raise ValueError("This is only reached if DRAW happens downstream from BLACK_WINS. This should not be possible!")
        p = []
    elif _board.bid == RED_WINS:
        if root_board.bid == BLACK_TO_MOVE or root_board.bid == BLACK_WINS:
            v = -1
        elif root_board.bid == RED_TO_MOVE or root_board.bid == RED_WINS:
            v = 1
        else:
            raise ValueError("This is only reached if DRAW happens downstream from RED_WINS. This should not be possible!")
        p = []
    else:
        v = 0
        _moves = _board.moves
        _len = len(_moves)
        p = list(mcts_node(_move, P = 1 / _len) for _move in _moves)

    return {
        "value": v,
        "policy": p
    }

# The number of games for a training network to play against itself before retraining and evaluation
_SELF_PLAY_GAMES = 256

# The number of games that will be written to disk for future training sessions
_GAMES_STORED = 1024

# The number of boards to be sampled from the _GAMES_STORED games for weight training
_SAMPLE_SIZE = 256

# The number of times to randomly sample games and train before evaluation
_TRAINING_LOOPS = 64

# The number of games to play against the best network to evaluate a newly trained network
_EVALUTATION_GAMES = 64

# The percentage of games that must be won by a new network to be declared the new best network
_WIN_THRESHOLD = 0.55

# The relative directory to store games
_GAMES_DIR = "connect_4\\games\\"

# This is the policy used for terminal boards. There isn't a clear choice here, so we make a global for ease of edit.
class _NULL_POLICY:
    as_dict = {
        PLACE_1: 1 / 7,
        PLACE_2: 1 / 7,
        PLACE_3: 1 / 7,
        PLACE_4: 1 / 7,
        PLACE_5: 1 / 7,
        PLACE_6: 1 / 7,
        PLACE_7: 1 / 7
    }

    as_array = np.full((7,), 1 / 7)

def _default_loss(y_true, y_pred):
        mse = MeanSquaredError()
        cat = CategoricalCrossentropy()
        return mse(y_true[:,0], y_pred[:,0]) + cat(y_true[:,1:], y_pred[:,1:])

def _convolutional_layer(inputs):
    outputs = Conv2D(32, (3,3), padding='same', kernel_regularizer = L2(), bias_regularizer = L2())(inputs)
    outputs = BatchNormalization()(outputs)
    return relu(outputs)

def _residual_layer(inputs):
    outputs = Conv2D(32, (3,3), padding='same', kernel_regularizer = L2(), bias_regularizer = L2())(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = relu(outputs)

    outputs = Conv2D(32, (3,3), padding='same', kernel_regularizer = L2(), bias_regularizer = L2())(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Add()([inputs,outputs])
    return relu(outputs)

class connect_4_model(ABC):
    @classmethod
    @abstractmethod
    def _get_model(cls):
        """
        This is the method that connect_4_model classes must override to define their model structure.
        The base class takes care of essentially all other logic.
        """
        ...
    
    @classmethod
    @property
    @abstractmethod
    def _model_dir(self):
        """
        This returns the directory to save the model in.
        Must override in base class.
        """
        ...

    def __init__(self, model: Model = None):
        if model is not None:
            self._model = model
        else:
            try:
                self.load()
            except IOError:
                self._model = self._get_model()
    
    def __call__(self, _board: connect_4_board):
        _input = np.empty((1, 6, 7, 2))
        _input[0,:,:,0] = _board.grid
        _input[0,:,:,1] = _board.bid.as_array
        r = self._model(_input)
        _policy = r[:,1:]

        return {
            "value": r[0,0].numpy(),
            "policy": {
                PLACE_1: _policy[0, PLACE_1.index].numpy(),
                PLACE_2: _policy[0, PLACE_2.index].numpy(),
                PLACE_3: _policy[0, PLACE_3.index].numpy(),
                PLACE_4: _policy[0, PLACE_4.index].numpy(),
                PLACE_5: _policy[0, PLACE_5.index].numpy(),
                PLACE_6: _policy[0, PLACE_6.index].numpy(),
                PLACE_7: _policy[0, PLACE_7.index].numpy()
            }
        }
    
    def fit(self, **kwargs):
        return self._model.fit(**kwargs)
    
    def copy(self) -> 'connect_4_model':
        model_copy= clone_model(self._model)
        model_copy.build((None, 6, 7, 2))
        model_copy.compile(optimizer='rmsprop', loss=_default_loss)
        model_copy.set_weights(self._model.get_weights())
        return type(self)(model = model_copy)
    
    def save(self):
        self._model.save(self._model_dir)
    
    def decision_func(self, root_board: connect_4_board, _board: connect_4_board):
        # We help the neural net learn by allowing the mcts to know about terminal boards
        # By having this first check for terminal boards, the network will learn it's policy faster... at least in theory
        if _board.bid == BLACK_WINS:
            if root_board.bid == BLACK_TO_MOVE or root_board.bid == RED_WINS or root_board.bid == DRAW:
                v = 1
            else:
                v = -1
            return {
                "value": v,
                "policy": []
            }
        elif _board.bid == RED_WINS:
            if root_board.bid == BLACK_TO_MOVE or root_board.bid == RED_WINS or root_board.bid == DRAW:
                v = -1
            else:
                v = 1
            return {
                "value": v,
                "policy": []
            }
        elif _board.bid == DRAW:
            v = 0
            return {
                "value": v,
                "policy": []
            }
        
        # Call the model on the virtual board
        r = self(_board)
        value_head = r["value"]
        policy_head = r["policy"]

        # Process the value for the current board state
        # The first if means it is black's turn, so the network outputs their true value
        # If it's not black's turn, then negate the value so it's from the persepctive of red
        # Technically the draw one is ambiguous...
        if root_board.bid == BLACK_TO_MOVE or root_board.bid == RED_WINS or root_board.bid == DRAW:
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

    @classmethod
    def load(cls) -> 'connect_4_model':
        return cls(model = load_model(cls._model_dir, custom_objects={'_default_loss': _default_loss}))
    
class connect_4_model_00(connect_4_model):
    @classmethod
    def _get_model(cls):
        # Inputs
        _input = Input(shape = (6, 7, 2))

        # Layers for board convolution embedding
        _internal = _convolutional_layer(_input)
        _internal = _residual_layer(_internal)
        _internal = Flatten()(_internal)

        # Layers after concatenation
        _internal = Dense(32, kernel_regularizer = L2(), bias_regularizer = L2())(_internal)
        _internal = Dense(16, kernel_regularizer = L2(), bias_regularizer = L2())(_internal)
        _policy_head = Dense(7, kernel_regularizer = L2(), bias_regularizer = L2(), activation = 'softmax')(_internal)
        _value_head = Dense(1, kernel_regularizer = L2(), bias_regularizer = L2(), activation='tanh')(_internal)
        _output = Concatenate()([_value_head, _policy_head])

        _model = Model(inputs = _input, outputs = _output)
        _model.build((None, 6, 7, 2))
        _model.compile(loss=_default_loss, optimizer = 'adam')

        return _model
    
    @classmethod
    @property
    def _model_dir(cls):
        return "connect_4\\models\\connect_4_model_00"

class mcts_trainer():
    def __init__(self, model: connect_4_model, game_pool_name: str = "best") -> None:
        """
        game_pool: This is a name for the pool of games the trainer will use for training.
            If no game_pool.json file exists in _GAMES_DIR, then a new one will be made.
        """
        # Start by saving the model and games
        self.model = model
        self.game_pool_name = game_pool_name
        try:
            with open(_GAMES_DIR + game_pool_name + '.json', "r") as f:
                self.game_pool = json.load(f)
        except FileNotFoundError:
            self.game_pool = {
                "games": []
            }
        
        # Deserialize the boards
        for _game in self.game_pool['games']:
            for b_and_p in _game['b_and_p']:
                b_and_p["board"] = connect_4_board.dejsonify(b_and_p["board"])
                b_and_p['policy'] = np.array(b_and_p['policy'])
            _game['value'] = np.array(_game['value'])
    
    def save_games(self):
        jsonified = list()
        for game in self.game_pool['games']:
            jsonified.append({
                'b_and_p': [{
                    'board': b_and_p['board'].jsonify(),
                    'policy': b_and_p['policy'].tolist()
                } for b_and_p in game['b_and_p']],
                'value': game['value'].tolist()
            })
        with open(_GAMES_DIR + self.game_pool_name + '.json', 'w') as f:
            f.truncate(0)
            json.dump({
                'games': jsonified
            }, f)
    
    def self_play(self, rounds: int = _SELF_PLAY_GAMES):
        for _ in range(rounds):
            # We being by initializing two instances of mcts on a shared board
            b = connect_4_board()
            m1 = mcts(b, self.model.decision_func)
            m2 = mcts(b, self.model.decision_func)

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
                _dict = current_mcts.move_stochastically()

                _move = _dict['move']
                all_nodes = _dict['all_nodes'] # This is a list of all the nodes that were possible moves, with true probabilities attached.

                # Now that the mcts has finished, we want to add the true probabilities to training data
                p = np.zeros((7,), dtype = float) # start by setting all probabilities equal to zero
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
                current_mcts.inform(_move)
            
            # Now the game is over, so we append the final board
            # There is not a clear policy to choose for the final board,
            # So we have globalized it to be consistent.
            game['b_and_p'].append({
                'board': b.copy(),
                'policy': _NULL_POLICY.as_array
            })

            # Once the game is over, we want to record the result
            # We store it in a numpy array because the neural network prefers this
            # When we serialize things will be stored differently
            if b.bid == BLACK_WINS:
                game['value'] = np.array([1.0])
            elif b.bid == RED_WINS:
                game['value'] = np.array([-1.0])
            else:
                game['value'] = np.array([0.0])
            
            logging.info("Game {0}/{1} of self play has finished.".format(_ + 1, rounds))

            self.game_pool['games'].append(game)

        # Once all the games have been played, prune the game pool to only include the desired history
        self.game_pool['games'] = self.game_pool['games'][-_GAMES_STORED:]
    
    def train_and_eval(self,
        sample_size: int = _SAMPLE_SIZE,
        training_loops: int = _TRAINING_LOOPS,
        batch_size: int = _SAMPLE_SIZE,
        epochs: int = 64,
        eval_games: int = _EVALUTATION_GAMES,
        thresh: float = _WIN_THRESHOLD):

        # In the train and eval step, we copy the best model, train it, and play the copy against the best
        trainee = self.model.copy()

        for _ in range(training_loops):
            # Pick our random games and set up the training data
            x_train = np.empty((sample_size, 6, 7, 2))
            y_train = np.empty((sample_size, 8))
            training_games = np.random.choice(self.game_pool['games'], sample_size)
            for i, game in enumerate(training_games):
                _dict = np.random.choice(game['b_and_p'])
                x_train[i, :, :, 0] = _dict['board'].grid
                x_train[i, :, :, 1] = _dict['board'].bid.as_array
                y_train[i, 0] = game['value']
                y_train[i, 1:] = _dict['policy']
            
            # Actually train the model on this data
            trainee.fit(
                x = x_train,
                y = y_train,
                batch_size = batch_size,
                epochs = epochs,
                verbose = 0)
            logging.info("Training loop {0}/{1} has finished.".format(_ + 1, training_loops))
        
        # Now play the trainee against the best model
        b = connect_4_board()
        trainee_player = mcts_player(b, trainee.decision_func)
        best_player = mcts_player(b, self.model.decision_func)

        trainee_player.wins = 0
        best_player.wins = 0
        draws = 0

        for n in range(eval_games):
            b.reset()
            trainee_player.reset()
            best_player.reset()

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
                current_player = next(it)

                # Inform it of the previous move that was made
                current_player.inform(_move)
            # Once the game is over, we want to record the result
            # When we serialize things will be stored differently
            if b.bid == BLACK_WINS:
                black_player.wins += 1
            elif b.bid == RED_WINS:
                red_player.wins += 1
            else:
                draws += 1

            logging.info("Game {0}/{1} of evaluation has finished.".format(n + 1, eval_games))
        
        # Now we check to see if the trainee player won enough games and declare it best if so
        logging.info("Trainee: " + str(trainee_player.wins))
        logging.info("Best: " + str(best_player.wins))
        logging.info("Draws: " + str(draws))
        if trainee_player.wins / eval_games > thresh:
            self.model = trainee
            logging.info("New best model!")
        
        self.model.save()