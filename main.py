from tree import *
from connect_4.implementations import *
from connect_4.train import *

t = mcts_trainer(connect_4_model())

t.self_play(1)

t.train_and_eval()

t.save_games()