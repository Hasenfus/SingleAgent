import gym
import numpy as np

alpha = 0.1  # learning rate
gamma = 0.99  # discount factor
epsilon = 1.0  # initial exploration rate
epsilon_min = 0.01  # minimum exploration rate
epsilon_decay = 0.995  # decay rate for exploration
n_episodes = 5000  # number of training episodes


env = gym.make('FrozenLake-v1', desc=["SFFF", "FHFH", "FFFH", "HFFG"])

Q = np.zeros([ env.observation_space.n,env.action_space.n])

for episode in range(n_episodes):
    state, _ = env.reset()
    done = False
    while not done:
        # Epsilon-greedy action selection
        if np.random.random_sample() < epsilon:
            action = np.random.randint(env.action_space.n)
        else:
            action = np.argmax(Q[state, :])

        next_state, reward, done, _ ,_  = env.step(action)
        
        # Q-value update
        Q[state, action] += alpha * (reward + gamma * max(Q[next_state, :]) - Q[state, action])
        
        state = next_state

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

n_eval_episodes = 1000
total_rewards = 0

for episode in range(n_eval_episodes):
    state,_ = env.reset()
    done = False
    while not done:
        action = np.argmax(Q[state, :])
        next_state, reward, done,_,_ = env.step(action)
        state = next_state
        total_rewards += reward

average_reward = total_rewards / n_eval_episodes
