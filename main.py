from tree import *
from connect_4.implementations import *
from connect_4.train import naive_decision_function

b = connect_4_board()
p2 = cli_player()
p1 = mcts_player(naive_decision_function, b.copy(b))

while not b.is_terminal:
    _move = p1.move(b)
    b = b.move(b, _move)
    p2.inform(_move)
    if b.is_terminal: break
    _move = p2.move(b)
    b = b.move(b, _move)
    p1.inform(_move)

print(b)