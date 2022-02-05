from connect_4.train import mcts_trainer

t = mcts_trainer()
t.self_play(50)
t.train_and_eval()
t.save_games()

# from connect_4.implementations import cli_player, random_player, connect_4_game, connect_4_board, naive_decision_func
# from tree import minimax_player
# from connect_4.train import connect_4_model, mcts_trainer, mcts_player

# m = connect_4_model.load('best')
# d = mcts_trainer.gen_decision_func(m)

# b = connect_4_board()
# p1 = mcts_player(b, d)
# p2 = minimax_player(b, naive_decision_func, 2)
# g = connect_4_game(b, p1, p2)
# r = g.play_tourney(100)
# print('P1 elo: ', p1.elo)
# print('P2 elo: ', p2.elo)
# print('Tourney resuts: ', r)