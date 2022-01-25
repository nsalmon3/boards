from typing import Callable

import numpy as np

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

class node_mcts(node):
    def __init__(self, parent = None, children = None, N = 0, W = 0, Q = 0, P = 0, **kwargs):
        super().__init__(parent, children, N = N, W = W, Q = Q, P = P, **kwargs)

_DEFAULT_PROPS = 100
_DEFAULT_TEMPERATURE = 0.5

class mcts():
    def __init__(self,
        decision_func: Callable[[node_mcts], dict],
        root: node_mcts,
        U: Callable[[node_mcts], float]):
        """
        decision_func:
            This takes in a node, specifically a leaf. It outputs a dictionary with the following format:
            {
                "value": "float in [-1,1], indicating value of this node",
                "policy": [
                    node1,
                    node2,
                    ...
                    noden
                ]
            }.
        
        root:
            This is the starting point for the tree search.
        """

        # Begin by initializing
        self.decision_func = decision_func
        self.root = root
        self.U = U
    
    def _propogate(self, n: int = _DEFAULT_PROPS):
        for i in range(n):
            # We start at the root node, and then choose nodes that maximize Q + U until a leaf node is reached
            _node = self.root
            while not _node.is_leaf():
                _max = 0
                _max_node = None
                for child in _node.children:
                    if child.Q + self.U(child) >= _max:
                        _max = child.Q + self.U(child)
                        _max_node = child
                _node = _max_node

            # Now we as the decision_func what it thinks about this leaf node
            _dict = self.decision_func(_node)

            # We add the new leaf nodes who will have their initialized probabilities
            _node.add(*_dict["policy"])

            # Now we must propogate the value back through the tree
            v = _dict["value"]
            while not _node.is_root():
                _node.N += 1
                _node.W += v
                _node.Q = _node.W / _node.N
                _node = _node.parent
    
    def run_deterministically(self, n: int = _DEFAULT_PROPS):
        self._propogate(n)


        ### QUESTIONABLE BEHAVIOR HERE ###
        # If the root node has no children after propogation, that means it is a true leaf in the tree
        # As it stands, we will simply return this node after being asked to make a move here.
        # Might be better to define a custom exception to throw instead.
        if self.root.children == []:
            return self.root

        _max = 0
        _max_node = None
        for child in self.root.children:
            if child.Q + self.U(child) > _max:
                _max = child.Q + self.U(child)
                _max_node = child
        
        self.root = _max_node.prune()

        return self.root

    def run_stochastically(self, n: int = _DEFAULT_PROPS, tau: float = _DEFAULT_TEMPERATURE):
        self._propogate(n)

        ### QUESTIONABLE BEHAVIOR HERE ###
        # If the root node has no children after propogation, that means it is a true leaf in the tree
        # As it stands, we will simply return this node after being asked to make a move here.
        # Might be better to define a custom exception to throw instead.
        if self.root.children == []:
            return self.root

        self.root = np.random.choice(self.root.children, p = [child.P for child in self.root.children]).prune()

        return self.root