import minesweeper as ms
import numpy as np 
import time

env = ms.MinesweeperEnv()
obs = env.reset()

NUM_GAMES = 10000
outcomes = []
rewards = []
curr_game_iter = 0

while curr_game_iter < NUM_GAMES:
    random_action = env.action_space.sample()
    while not env.valid_actions[random_action[0]][random_action[1]]:
        random_action = env.action_space.sample()
    new_obs, reward, done, info = env.step(random_action)

    if done:
        curr_game_iter += 1
        outcomes.append(ms.is_win(new_obs))
        rewards.append(reward)
        obs = env.reset()

print("Strategy: Random")
print("Number of games played: {}".format(len(outcomes)))
print("Win rate: {}".format(np.mean(outcomes)))
print("Average reward: {}".format(np.mean(rewards)))

env.close()
