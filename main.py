from tree import *
from connect_4.implementations import *
from connect_4.train import *

t = mcts_trainer()

t.self_play(500)

t.train_and_eval()

t.save_games()