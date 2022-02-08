from connect_4.implementations import BLACK_WINS, RED_WINS, connect_4_board, naive_value_func
from connect_4.train import _NULL_POLICY, _GAMES_DIR
from tree import minimax_player

from itertools import cycle
import numpy as np
import json

def main(b: connect_4_board, p1, p2, rounds: int = 10, reset = False):
    """
    This script is designed to fill a game pool with games from given players, with the intention of later training on these games.
    """
    game_pool = list()

    for n in range(rounds):
        b.reset()

        game = {
            'b_and_p': list(),
            'value': None
        }

        it = iter(cycle([p1, p2]))
        current_player = next(it)
        while not b.is_terminal:
            _move = current_player.move()
            p = [0] * 7
            p[_move.index] = 1.

            game['b_and_p'].append({
                'board': b.copy().jsonify(),
                'policy': p
            })

            b.move(_move)

            current_player = next(it)
            current_player.inform(_move)
        
        game['b_and_p'].append({
            'board': b.copy().jsonify(),
            'policy': list(_NULL_POLICY.as_array)
        })

        if b.bid == BLACK_WINS:
            game['value'] = [1.0]
        elif b.bid == RED_WINS:
            game['value'] = [-1.0]
        else:
            game['value'] = [0.0]
        
        print('Game {0}/{1} of bootstrapping has finished.'.format(n + 1, rounds))

        game_pool.append(game)

    if not reset:
        with open(_GAMES_DIR + "best.json", 'r') as f:
            game_dict = json.load(f)
        for game in game_pool:
            game_dict['games'].append(game)
        with open(_GAMES_DIR + 'best.json', 'w') as f:
            f.truncate(0)
            json.dump(game_dict, f)
    else:
        with open(_GAMES_DIR + "best.json", 'w') as f:
            f.truncate(0)
            json.dump({
                'games': game_pool
            }, f)