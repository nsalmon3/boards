import sys

sys.path.append("..\\boards")

from connect_4.implementations import *
from connect_4.nn.nn import *
from tree import mcts_node

def naive_decision_function(_board: connect_4_board):
    if _board.id == 'BW': 
        v = 1
        p = []
    elif _board.id == 'RW':
        v = -1
        p = []
    else:
        v = 0
        _moves = _board.moves
        if 'OD' in _moves: _moves.remove('OD')
        if 'DD' in _moves: _moves.remove('DD')
        if 'R' in _moves: _moves.remove('R')
        _len = len(_moves)
        p = list(mcts_node(_move, P = 1 / _len) for _move in _moves)

    return {
        "value": v,
        "policy": p
    }