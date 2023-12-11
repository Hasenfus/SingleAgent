"""Tabular QL agent"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils
import gymnasium as gym
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import random

DEBUG = False

# GAMMA = 0.95  # discounted factor
# TRAINING_EP = 0.25  # epsilon-greedy parameter for training
TESTING_EP = 0.05  # epsilon-greedy parameter for testing
NUM_RUNS = 10
NUM_EPOCHS = 200
NUM_EPIS_TRAIN = 25  # number of episodes for training at each epoch
NUM_EPIS_TEST = 50 # number of episodes for testing
# ALPHA = 0.8  # learning rate for training

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def index2tuple(index, n):
    """Converts an index c to a tuple (a,b)"""
    return index // n, index % n

# pragma: coderesponse template
def epsilon_greedy(s, q_func, epsilon):
    """Returns an action selected by an epsilon-Greedy exploration policy

    Args:
        state_1, state_2 (int, int): two indices describing the current state
        q_func (np.ndarray): current Q-function
        epsilon (float): the probability of choosing a random command

    Returns:
        (int, int): the indices describing the action/object to take
    """
    coin = np.random.random_sample()
    if coin < epsilon:
        action_index = np.random.randint(0,NUM_ACTIONS)
    else:
        q_values = q_func[s, :]
        action_index = np.unravel_index(np.argmax(q_values, axis=None),
                                          q_values.shape)[0]
    return action_index


# pragma: coderesponse end


# pragma: coderesponse template
def tabular_q_learning(q_func):
    """Update q_func for a given transition

    Args:
        q_func (np.ndarray): current Q-function
        current_state_1, current_state_2 (int, int): two indices describing the current state
        action_index (int): index of the current action
        object_index (int): index of the current object
        reward (float): the immediate reward the agent recieves from playing current command
        next_state_1, next_state_2 (int, int): two indices describing the next state
        terminal (bool): True if this episode is over

    Returns:
        None
    """
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
    non_final_next_states = np.array([s for s in batch.next_state if s is not None])

    state_batch = np.array(batch.state)
    action_batch = np.array(batch.action)
    reward_batch = np.array(batch.reward)

    state_action_values = q_func[state_batch, action_batch]

    next_state_values = np.zeros(BATCH_SIZE)
    next_state_values[non_final_mask] = q_func[non_final_next_states, :].max(1)[0]

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    q_func[state_batch, action_batch] = (1-ALPHA) * state_action_values + ALPHA * expected_state_action_values

    return None


# pragma: coderesponse end


# pragma: coderesponse template
def run_episode(for_training):
    """ Runs one episode
    If for training, update Q function
    If for testing, computes and return cumulative discounted reward

    Args:
        for_training (bool): True if for training

    Returns:
        None
    """
    epsilon = TRAINING_EP if for_training else TESTING_EP

    epi_reward = 0
    # initialize for each episode
    gamma_step = 1
    s, _ = env.reset()
    terminal = False

    # print(current_room_desc,current_quest_desc)
    step_count = 0
    while not terminal:
        # Choose next action and execute

        a = epsilon_greedy(s, q_func, epsilon)
        next_s, reward, terminal, _, _ = env.step(a)
        if terminal:
            next_s = None
        else:
            next_s = torch.tensor(next_s)
        
        memory.push(s,a,next_s,reward)

        if for_training:
            # update Q-function.
            tabular_q_learning(q_func)
            pass

        if not for_training:
            epi_reward = epi_reward + gamma_step * reward
            gamma_step = gamma_step * GAMMA
            

        # prepare next step
        s = next_s
        step_count += 1
    if not for_training:
        return epi_reward, step_count
    


# pragma: coderesponse end


def run_epoch():
    """Runs one epoch and returns reward averaged over test episodes"""
    rewards = []
    durations = []

    for _ in range(NUM_EPIS_TRAIN):
        run_episode(for_training=True)

    for _ in range(NUM_EPIS_TEST):
        reward, duration = run_episode(for_training=False)
        rewards.append(reward)
        durations.append(duration)

    return np.mean(np.array(rewards)), np.mean(np.array(durations))


def run():
    """Returns array of test reward per epoch for one run"""
    global q_func
    q_func = np.zeros((NUM_STATES, NUM_ACTIONS))

    single_run_epoch_rewards_test = []
    single_run_epoch_durations_test = []
    pbar = tqdm(range(NUM_EPOCHS), ncols=80)
    for _ in pbar:
        rewards, durations = run_epoch()
        single_run_epoch_rewards_test.append(rewards)
        single_run_epoch_durations_test.append(durations)
        pbar.set_description(
            "Avg reward: {:0.6f} | Ewma reward: {:0.6f}".format(
                np.mean(single_run_epoch_rewards_test),
                utils.ewma(single_run_epoch_rewards_test)))
    return single_run_epoch_rewards_test, q_func, single_run_epoch_durations_test


if __name__ == '__main__':
    # Data loading and build the dictionaries that use unique index for each state
    # (dict_room_desc, dict_quest_desc) = framework.make_all_states_index()
    # NUM_ROOM_DESC = len(dict_room_desc)
    # NUM_QUESTS = len(dict_quest_desc)

    # set up the game
    # framework.load_game_data()
    agent_eps = [0.5]
    agent_alpha = [0.1]
    agent_gamma = [0.95]
    agent_ = []
    global TRAINING_EP
    global ALPHA
    global GAMMA
    global BATCH_SIZE
    BATCH_SIZE = 32

    memory = ReplayMemory(1000)

    desc=["SFFF", "FHFH", "FFFH", "HFFG"]
    global env 
    env = gym.make('FrozenLake-v1', desc=desc, map_name="4x4", is_slippery=True)
    NUM_ACTIONS = env.action_space.n
    NUM_STATES = env.observation_space.n
    for TRAINING_EP in agent_eps:
        for ALPHA in agent_alpha:
            for GAMMA in agent_gamma:
                epoch_rewards_test = []  # shape NUM_RUNS * NUM_EPOCHS
                durations_test = []
                print("Training epsilon: ", TRAINING_EP)
                print("Alpha: ", ALPHA)
                print("Gamma: ", GAMMA)

                for _ in range(NUM_RUNS):
                    rewards, q_func, durations = run()
                    epoch_rewards_test.append(rewards)
                    durations_test.append(durations)

                durations = np.array(durations_test)
                epoch_rewards_test = np.array(epoch_rewards_test)

                x = np.arange(NUM_EPOCHS)
                agent_.append((TRAINING_EP,ALPHA,GAMMA, np.mean(epoch_rewards_test,axis=0), q_func, np.mean(durations,axis=0)))
    rewards_test = [i[3] for i in agent_]
    i = np.argmax(rewards_test) // NUM_EPOCHS
    np.save('results/2q_func' + str(i) + '.npy', np.array(agent_[i][4]),)

    fig, axis = plt.subplots()
    for i in agent_:
        axis.plot(x, i[3], label=('Epsilon=%.2f, alpha=%.4f, location=%.4f' % (i[0], i[1], i[2])))  # plot reward per epoch averaged per run
    axis.set_xlabel('Epochs')
    axis.set_ylabel('reward')
    axis.legend()
    axis.set_title(('Tablular: nRuns=%d, Epilon=%.2f, Epi=%d, alpha=%.4f' %
                    (NUM_RUNS, TRAINING_EP, NUM_EPIS_TRAIN, ALPHA)))
    plt.savefig('runs/2tabular.png')

    axis.clear()
    for i in agent_:
        axis.plot(x, i[5], label=('Epsilon=%.2f, alpha=%.4f, location=%.4f' % (i[0], i[1], i[2])))
    axis.set_xlabel('Epochs')
    axis.set_ylabel('duration')
    axis.legend()
    axis.set_title(('Tablular: nRuns=%d, Epilon=%.2f, Epi=%d, alpha=%.4f' %
                    (NUM_RUNS, TRAINING_EP, NUM_EPIS_TRAIN, ALPHA)))
    plt.savefig('runs/2tabular_duration.png')