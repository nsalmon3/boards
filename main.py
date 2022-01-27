from tree import *
from connect_4.implementations import *
from connect_4.train import *

b = connect_4_board()

t = mcts_trainer(connect_4_model())

t.training_round()