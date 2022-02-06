from connect_4.implementations import cli_player, random_player, connect_4_game, connect_4_board, naive_value_func
from tree import minimax_player
from connect_4.train import connect_4_model_00, mcts_trainer, mcts_player

# First select the desired model to train
m = connect_4_model_00()

# Initialize the trainer to train this model
t = mcts_trainer(model = m)

# One loop is self play, fitting, and evaluation
t.self_play(512)
t.train_and_eval()

# Save the games
# Might also move this to the __del__ method of the trainer...?
t.save_games()

# Here we test the player against a naive minimax player.
# After 100 games we see how the model did.
b = connect_4_board()
p1 = mcts_player(b, m.decision_func)
p2 = minimax_player(b, naive_value_func, 2)
g = connect_4_game(b, p1, p2)
r = g.play_tourney(100)
print('P1 elo: ', p1.elo)
print('P2 elo: ', p2.elo)
print('Tourney resuts: ', r)