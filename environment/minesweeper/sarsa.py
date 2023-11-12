import numpy as np
import minesweeper as ms
import sys
import time

gamma = 0.9
alpha = 0.15
iterations = int(1e6)
epsilon = 1

def greedy_exploration(env, action_space, action_vals, epsilon):
    rand = np.random.uniform(0, 1)
    if rand <= epsilon:
        action = action_space.sample()
        while env.valid_actions[action[0]][action[1]] == 0:
            action = action_space.sample()
        return action 
    else:
        return np.unravel_index(action_vals.argmax(), action_vals.shape)

def run():
    print("Starting run")
    env = ms.MinesweeperEnv()
    action_space = env.action_space
    state = env.reset() 
    Q_vals = {}

    for i in range(iterations):
        if i % 1000 == 0:
            print("run: ", i)
            sys.stdout.flush()
        state_str = ms.board2str(state)
        if state_str not in Q_vals.keys():
            Q_vals[state_str] = np.zeros((ms.BOARD_SIZE, ms.BOARD_SIZE))
        action_vals = np.copy(Q_vals[state_str])
        action_vals[state != -2] = -float('inf')

        curr_action = greedy_exploration(env, action_space, action_vals, epsilon)

        done = False 
        while not done:
            next_state, reward, done, _ = env.step((curr_action[0], curr_action[1]))

            next_action = greedy_exploration(env, action_space, action_vals, epsilon)
            q_val = Q_vals[state_str][curr_action[0]][curr_action[1]]
            q_val_next = Q_vals[state_str][next_action[0]][next_action[1]]
            update = reward + (gamma*(q_val_next-q_val))
            Q_vals[state_str][curr_action[0]][curr_action[1]] = q_val + alpha*update 

            if done:
                state = env.reset()
            else:
                state = next_state
                state_str = ms.board2str(state)
                curr_action = next_action
                if state_str not in Q_vals.keys():
                    Q_vals[state_str] = np.zeros((ms.BOARD_SIZE, ms.BOARD_SIZE))
                action_vals = np.copy(Q_vals[state_str])
                action_vals[state != -2] = -float('inf')
    return Q_vals


def eval(env, Q_vals):
    print("Starting eval")
    num_games = 10000
    rewards = []
    results = []

    for i in range(num_games):
        if i%1000 == 0:
            print("eval: ", i)
            sys.stdout.flush()

        totalReward = 0
        done = False 
        state = env.reset() 
        while not done:
            state_str = ms.board2str(state)
            if state_str in Q_vals.keys():
                action_vals = np.copy(Q_vals[state_str])
                action_vals[state != -2] = -float('inf')
                action = np.unravel_index(action_vals.argmax(), action_vals.shape)
            else:
                action = env.action_space.sample()
                while not env.valid_actions[action[0]][action[1]]:
                    action = env.action_space.sample()

            state, reward, done, _ = env.step(action)
            totalReward += reward

            if done:
                results.append(ms.is_win(state))
                rewards.append(totalReward)

    return results, rewards

start_time = time.time()
Q_vals = run()
runtime = time.time()-start_time
env = ms.MinesweeperEnv()
results, rewards = eval(env, Q_vals)
print(runtime, np.mean(results), np.mean(rewards))
