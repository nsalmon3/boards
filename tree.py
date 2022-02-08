from typing import Callable
import numpy as np

from abstract import *
from connect_4.implementations import BLACK_TO_MOVE, RED_TO_MOVE

class node():
    def __init__(self, parent = None, children = None, **kwargs):
        self.children = children
        self.parent = parent
        for arg in kwargs:
            self.__dict__[arg] = kwargs[arg]

    def __str__(self):
        return str(vars(self))

    def add(self, *_nodes):
        if self.children is None:
            self.children = list()
        if _nodes == None:
            return
        for _node in _nodes:
            if _node.parent is not None:
                raise ValueError("Attempted to add a node which already has a parent!")
            _node.parent = self
            self.children.append(_node)

    def is_leaf(self):
        return self.children is None or len(self.children) == 0

    def is_root(self):
        return self.parent == None

    def prune(self):
        self.parent = None
        return self

    def remove(self, *_nodes):
        if self.children is None:
            raise Exception("Tried to remove nodes from node with no children!")
        for _node in _nodes:
            if _node not in self.children:
                raise ValueError("Attempted to remove a node which is not a child!")
            self.children.remove(_node)

class mcts_node(node):
    def __init__(self,
        _move = None,
        parent: 'mcts_node' = None,
        children: 'list[mcts_node]' = None,
        N: int = 0,
        W: float = 0,
        Q: float = 0, 
        P: float = 0,
        v: float = 0,
        **kwargs):

        super().__init__(parent, children, move = _move, N = N, W = W, Q = Q, P = P, v = v, **kwargs)

_DEFAULT_PROPS = 7 ** 3
_TEMPERATURE = 1
_EXPLORATION_CONSTANT = 1

class mcts():
    def __init__(self,
        _board: board,
        decision_func: Callable[[board, board], dict]):
        """
        decision_func:
            This takes in a root board and a board to examine, specifically at a leaf. It outputs a dictionary with the following format:
            {
                "value": "float in [-1,1], indicating value of this node",
                "policy": [
                    mcts_node_1,
                    mcts_node_2,
                    ...
                    mcts_node_n
                ]
            }
        The nodes returned by the policy will have their P populated appropriately, and this is how we access the policy.
        The idea is that a virtual board gets passed around as we traverse the tree.
        The root board tells the decision function from what perspective we are evaluating the virtual board from.
        This is mostly to obtain which player is evaluating the virtual board.
        """

        # Begin by initializing
        self.decision_func = decision_func
        self.board = _board
        self.root_node = mcts_node()
    
    def _propogate(self, n: int = _DEFAULT_PROPS):
        for _ in range(n):
            # We start at the root node, and then choose nodes that maximize Q + U until a leaf node is reached
            _node = self.root_node
            _board = self.board.copy()
            while not _node.is_leaf():
                _max = -np.Inf
                _max_nodes = list()
                _sqrt = np.sqrt(sum(_n.N for _n in _node.children))
                for child in _node.children:
                    _U = _EXPLORATION_CONSTANT * child.P * (1 + _sqrt) / (1 + child.N)
                    if child.Q + _U > _max:
                        _max = child.Q + _U
                        _max_nodes = [child]
                    elif child.Q + _U == _max:
                        _max_nodes.append(child)
                _node = np.random.choice(_max_nodes)
                _board.move(_node.move)

            # Now we as the decision_func what it thinks about this leaf node
            _dict = self.decision_func(self.board, _board)

            # We add the new leaf nodes who will have their initialized probabilities
            _node.add(*_dict["policy"])

            # Now we must propogate the value back through the tree
            v = _dict["value"]
            _node.v = v
            while not _node.is_root():
                _node.N += 1
                _node.W += v
                _node.Q = _node.W / _node.N
                _node = _node.parent
            # Now update the root
            _node.N += 1
            _node.W += v
            _node.Q = _node.W / _node.N
    
    def inform(self, _move):
        # If the root node is a leaf, then it has never been visited and we essentially start a new tree under it.
        if self.root_node.children is None or len(self.root_node.children) == 0:
            self.root_node = mcts_node()
            return
        
        # In this case the root node has already been expanded with all possible options, so we find the relevant move.
        for child in self.root_node.children:
            if child.move == _move:
                self.root_node = child
                return
        raise ValueError("mcts tried to make move " + _move.as_string + " on board:\n" + str(self.board))
    
    def move_deterministically(self, n: int = _DEFAULT_PROPS):
        """
        Runs looking to make the best move it can. Returns the move.
        """
        self._propogate(n)

        # If the root node has no children after propogation
        if self.root_node.children is None:
            return None

        _max = -np.Inf
        _max_nodes = list()
        for child in self.root_node.children:
            if child.N > _max:
                _max = child.N
                _max_nodes = [child]
            elif child.N == _max:
                _max_nodes.append(child)

        self.root_node = np.random.choice(_max_nodes).prune()

        return self.root_node.move
    
    def move_minimax_deterministically(self, n: int = _DEFAULT_PROPS):
        """
        Runs a mcts as normal. However, rather than pick the move with most visits,
        it does a minimax on Q to pick it's move.
        """
        self._propogate(n)

        # If the root node has no children after propogation
        if self.root_node.children is None:
            return None

        _max = -np.Inf
        _max_nodes = list()
        for child in self.root_node.children:
            val = mcts_minimax(child, False)
            if val > _max:
                _max = val
                _max_nodes = [child]
            elif val == _max:
                _max_nodes.append(child)

        self.root_node = np.random.choice(_max_nodes).prune()

        return self.root_node.move

    def move_stochastically(self, n: int = _DEFAULT_PROPS) -> dict:
        """
        Runs looking to explore and practice the game. Returns a dictionary of relevant training data for trainer.
        """
        self._propogate(n)

        if self.root_node.children is None:
            return {
                "move": None,
                "all_nodes": None
            }

        p = [child.N ** (1 / _TEMPERATURE) for child in self.root_node.children]
        p = list(map(lambda x: x / sum(p), p))
        all_nodes = self.root_node.children
        _sum = sum([_nodes.N ** (1 / _TEMPERATURE) for _nodes in all_nodes])
        self.root_node = np.random.choice(self.root_node.children, p = p).prune()

        for _node in all_nodes:
            _node.P = (_node.N ** (1 / _TEMPERATURE)) / _sum

        return {
            "move": self.root_node.move,
            "all_nodes": all_nodes
        }
    
    def reset(self):
        self.root_node = mcts_node()

class mcts_player(player):
    def __init__(self, _board: board, decision_func: Callable[[board, board], dict], _elo: elo = None):
        super().__init__(_board, _elo = _elo)
        self.mcts = mcts(_board, decision_func)

    def move(self) -> bid:
        return self.mcts.move_deterministically()
    
    def inform(self, _move: bid):
        self.mcts.inform(_move)

    def reset(self):
        self.mcts.reset()

def mcts_minimax(current_node: mcts_node, maximizing_player: bool):
    """
    The abstract minimax function. It wants to know who is asking and what board they are asking about.
    """

    # First check if we have reached our desired minimax depth or if the boardstate is a final state.
    if current_node.is_leaf():
        return current_node.v
    # Now is the logic for the minimax algorithm
    if maximizing_player:
        value = -np.Inf
        for child in current_node.children:
            value = max(value, mcts_minimax(child, False))
        return value
    else:
        value = np.Inf
        for child in current_node.children:
            value = min(value, mcts_minimax(child, True))
        return value

class minimax_mcts_player(player):
    def __init__(self, _board: board, decision_func: Callable[[board, board], dict], _elo: elo = None):
        super().__init__(_board, _elo = _elo)
        self.mcts = mcts(_board, decision_func)

    def move(self) -> bid:
        return self.mcts.move_minimax_deterministically()
    
    def inform(self, _move: bid):
        self.mcts.inform(_move)

    def reset(self):
        self.mcts.reset()

def minimax(root_board: board, _board: board, depth, maximizing_player: bool, decision_func: Callable[[board, board], float]):
    """
    The abstract minimax function. It wants to know who is asking and what board they are asking about.
    """

    # First check if we have reached our desired minimax depth or if the boardstate is a final state.
    if depth == 0 or _board.is_terminal:
        return decision_func(root_board, _board)

    # Now is the logic for the minimax algorithm
    if maximizing_player:
        value = -np.Inf
        for move in _board.moves:
            value = max(value, minimax(root_board, _board.move(move, inplace = False), depth - 1, False, decision_func))
        return value
    else:
        value = np.Inf
        for move in _board.moves:
            value = min(value, minimax(root_board, _board.move(move, inplace = False), depth - 1, True, decision_func))
        return value
 
class minimax_player(player):
    def __init__(self, _board: board, decision_func: Callable[[board, board], float], depth: int, _elo: elo = None) -> None:
        super().__init__(_board, _elo)
        self.decision_func = decision_func
        self.depth = depth
    
    def move(self) -> bid:
        best_moves = list()
        _min = -np.Inf
        for move in self.board.moves:
            val = minimax(self.board, self.board.move(move, inplace = False), self.depth - 1, False, self.decision_func)
            if val > _min:
                _min = val
                best_moves = [move]
            elif val == _min:
                best_moves.append(move)
        return np.random.choice(best_moves)
    
    def inform(self, _move: bid):
        ...
    
    def reset(self):
        ...