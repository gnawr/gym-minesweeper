import gymnasium as gym
from stable_baselines3 import DQN
import minesweeper as ms
import numpy as np

env = ms.MinesweeperDiscreetEnv()
new_obs = env.reset()

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=3000000, log_interval=4, progress_bar=True)
model.save("dqn")

# del model # remove to demonstrate saving and loading

# model = DQN.load("dqn_cartpole")

# obs, info = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         obs, info = env.reset()


NUM_GAMES = 10000
outcomes = []
rewards = []
curr_game_iter = 0
curr_game_reward = 0
step_i = 0

while curr_game_iter < NUM_GAMES:
    if step_i % 100 == 0:
        print("Step {}".format(step_i))
        print("Game {}".format(curr_game_iter))
    if step_i > 15: # prevent infinite loop
        break

    # call DQN
    action, _states = model.predict(new_obs, deterministic=True)
    print(f'Action: {action}')

    new_obs, reward, done, info = env.step(action)
    curr_game_reward += reward
    step_i += 1

    if done:
        outcomes.append(ms.is_win(new_obs))
        rewards.append(curr_game_reward)

        obs = env.reset()
        curr_game_iter += 1
        curr_game_reward = 0

print("Strategy: DQN")
print("Number of games played: {}".format(len(outcomes)))
print("Win rate: {}".format(np.mean(outcomes)))
print("Average reward: {}".format(np.mean(rewards)))

env.close()
