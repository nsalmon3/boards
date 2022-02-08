import tensorflow as tf
from connect_4.implementations import cli_player, random_player, connect_4_game, connect_4_board, naive_value_func, connect_4_round_robin
from tree import minimax_mcts_player, minimax_player
from connect_4.train import connect_4_model_00, mcts_trainer, mcts_player, naive_mcts_function
import connect_4.bootstrap as bs

b = connect_4_board()
m = connect_4_model_00()
t = mcts_trainer(m)
t.train_and_eval(eval_games=20)

p1 = mcts_player(b, m.decision_func)
p2 = minimax_player(b, naive_value_func, 4)

g = connect_4_game(b, p1, p2)
print(g.play(10))