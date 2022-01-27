from connect_4.implementations import *
from tree import mcts_node

def naive_decision_function(_board: connect_4_board):
    if _board.id == BLACK_WINS: 
        v = 1
        p = []
    elif _board.id == RED_WINS:
        v = -1
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