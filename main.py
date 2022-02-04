from connect_4.train import mcts_trainer

t = mcts_trainer()
t.self_play(500)
t.train_and_eval()
t.save_games()