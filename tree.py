from typing import Callable
from abstract import *

import numpy as np

from connect_4.implementations import connect_4_board

class node():
    def __init__(self, parent = None, children = None, **kwargs):
        if children is None:
            self.children = list()
        else:
            self.children = children
        self.parent = parent
        for arg in kwargs:
            self.__dict__[arg] = kwargs[arg]

    def __str__(self):
        return str(vars(self))

    def add(self, *_nodes):
        if _nodes == None:
            return
        elif len(_nodes) == 0:
            return
        for _node in _nodes:
            if _node.parent is not None:
                raise ValueError("Attempted to add a node which already has a parent!")
            _node.parent = self
            self.children.append(_node)

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent == None

    def prune(self):
        self.parent = None
        return self

    def remove(self, *_nodes):
        for _node in _nodes:
            if _node not in self.children:
                raise ValueError("Attempted to remove a node which is not a child!")
            self.children.remove(_node)

class mcts_node(node):
    def __init__(self,
        move: str = None,
        parent: 'mcts_node' = None,
        children: 'mcts_node' = None,
        N: int = 0,
        W: float = 0,
        Q: float = 0, 
        P: float = 0,
        **kwargs):

        super().__init__(parent, children, move = move, N = N, W = W, Q = Q, P = P, **kwargs)

_DEFAULT_PROPS = 1000
_DEFAULT_TEMPERATURE = 0.5
_EXPLORATION_CONSTANT = 1

class mcts():
    def __init__(self,
        decision_func: Callable[[board], dict],
        root_board: board):
        """
        decision_func:
            This takes in a board, specifically at a leaf. It outputs a dictionary with the following format:
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
        This way we do not have a board for each node, which would take up inconceivable amounts of memory.
        
        root:
            This is the starting point for the tree search.
        """

        # Begin by initializing
        self.decision_func = decision_func
        self.root_board = root_board
        self.root_node = mcts_node()
    
    def _propogate(self, n: int = _DEFAULT_PROPS):
        for _ in range(n):
            # We start at the root node, and then choose nodes that maximize Q + U until a leaf node is reached
            _node = self.root_node
            _board = self.root_board.copy(self.root_board)
            while not _node.is_leaf():
                _max = -np.Inf
                _max_node = None
                _sqrt = np.sqrt(sum(_n.N for _n in _node.children))
                for child in _node.children:
                    _U = _EXPLORATION_CONSTANT * child.P * _sqrt / (1 + child.N)
                    if child.Q + _U >= _max:
                        _max = child.Q + _U
                        _max_node = child
                _node = _max_node
                _board.move(_board, _node.move, True)

            # Now we as the decision_func what it thinks about this leaf node
            _dict = self.decision_func(_board)

            # We add the new leaf nodes who will have their initialized probabilities
            _node.add(*_dict["policy"])

            # Now we must propogate the value back through the tree
            v = _dict["value"]
            while not _node.is_root():
                _node.N += 1
                _node.W += v
                _node.Q = _node.W / _node.N
                _node = _node.parent
    
    def move(self, _move: str):
        # If a leaf node is the root node, then it has never been visited and we essentially start a new tree under it.
        if self.root_node.children == []:
            self.root_node = mcts_node()
            self.root_board.move(self.root_board, _move, inplace = True)
            return
        
        # In this case the root node has already been expanded with all possible options, so we find the relevant move.
        for child in self.root_node.children:
            if child.move == _move:
                self.root_node = child
                self.root_board = self.root_board.move(self.root_board, _move)
                return
        raise ValueError("mcts tried to make a move on a root board that wasn't an option... it's confused!")
    
    def run_deterministically(self, n: int = _DEFAULT_PROPS) -> str:
        self._propogate(n)


        ### QUESTIONABLE BEHAVIOR HERE ###
        # If the root node has no children after propogation, that means it is a true leaf in the tree
        # As it stands, we will simply return this node after being asked to make a move here.
        # Might be better to define a custom exception to throw instead.
        if self.root_node.children == []:
            return self.root_node

        _max = -np.Inf
        _max_node = None
        _sqrt = np.sqrt(sum(_n.N for _n in self.root_node.children))
        for child in self.root_node.children:
            _U = _EXPLORATION_CONSTANT * child.P * _sqrt / (1 + child.N)
            if child.Q + _U > _max:
                _max = child.Q + _U
                _max_node = child
        
        self.root_board = self.root_board.move(self.root_board, _max_node.move)
        self.root_node = _max_node.prune()

        return self.root_node.move

    def run_stochastically(self, n: int = _DEFAULT_PROPS, tau: float = _DEFAULT_TEMPERATURE) -> str:
        self._propogate(n)

        ### QUESTIONABLE BEHAVIOR HERE ###
        # If the root node has no children after propogation, that means it is a true leaf in the tree
        # As it stands, we will simply return this node after being asked to make a move here.
        # Might be better to define a custom exception to throw instead.
        if self.root_node.children == []:
            return self.root_node

        p = [child.N ** (1 / _DEFAULT_TEMPERATURE) for child in self.root_node.children]
        p = list(map(lambda x: x / sum(p), p))
        self.root_node = np.random.choice(self.root_node.children, p = p).prune()
        self.root_board = self.root_board.move(self.root_board, self.root_node.move)

        return self.root_node.move

class mcts_player(player):
    def __init__(self, decision_func, root_board):
        self.mcts = mcts(decision_func, root_board)

    def move(self, _board: board) -> str:
        return self.mcts.run_deterministically()
    
    def inform(self, _move: str):
        self.mcts.move(_move)