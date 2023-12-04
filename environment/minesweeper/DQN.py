import gymnasium as gym
from stable_baselines3 import DQN
import minesweeper as ms
import numpy as np
import time

env = ms.MinesweeperDiscreetEnv()
new_obs = env.reset()
NUM_TRAINING_STEPS = 3000000
gamma = 0.85
log_name = "dqn_train"+str(NUM_TRAINING_STEPS) + "_gamma" + str(gamma)


model = DQN("MlpPolicy", env,gamma=gamma, verbose=1)
start_time = time.perf_counter()
model.learn(total_timesteps=NUM_TRAINING_STEPS, tb_log_name=log_name, log_interval=4, progress_bar=True)
end_time = time.perf_counter()
model.save(log_name)

# del model # remove to demonstrate saving and loading

# model = DQN.load(log_name)

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
curr_game_step = 0

loss_due_to_same_action_counter = 0

global_step_i = 0

while curr_game_iter < NUM_GAMES:
    if global_step_i % 100 == 0:
        print("Step {}".format(global_step_i))
        print("Game {}".format(curr_game_iter))

    if curr_game_step > 40: # prevent infinite loop, call lost game
        print("done due to same action, loss")
        outcomes.append(False) # is not win
        rewards.append(curr_game_reward)

        obs = env.reset()
        curr_game_iter += 1
        curr_game_step = 0
        curr_game_reward = 0
        loss_due_to_same_action_counter += 1

    # call DQN
    action, _states = model.predict(new_obs, deterministic=True)
    new_obs, reward, done, info = env.step(action)
    curr_game_reward += reward
    curr_game_step += 1
    global_step_i += 1

    if done:
        print("done")
        outcomes.append(ms.is_win(new_obs))
        rewards.append(curr_game_reward)

        obs = env.reset()
        curr_game_iter += 1
        curr_game_step = 0
        curr_game_reward = 0

print("Strategy: DQN")
print("Number of games played: {}".format(len(outcomes)))
print("Win rate: {}".format(np.mean(outcomes)))
print("Average reward: {}".format(np.mean(rewards)))
print("Number of losses due to same action: {}".format(loss_due_to_same_action_counter))
print("Training time: {}".format(end_time - start_time))

env.close()
