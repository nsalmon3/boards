from tree import *
from connect_4.implementations import *
from connect_4.train import naive_decision_function
from abstract import sequential_game

b = connect_4_board()
g = sequential_game(b, mcts_player(naive_decision_function, b), random_player(b))

report = g.play()

for _board in report['boards']:
    print(connect_4_board.deserialize(_board))