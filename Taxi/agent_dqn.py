"""Tabular QL agent"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils
import gymnasium as gym
import random
from collections import namedtuple, deque
from itertools import count

DEBUG = False

GAMMA = 0.95  # discounted factor
TRAINING_EP = 0.5  # epsilon-greedy parameter for training
TESTING_EP = 0.05  # epsilon-greedy parameter for testing
NUM_RUNS = 5
NUM_EPOCHS = 200
NUM_EPIS_TRAIN = 25  # number of episodes for training at each epoch
NUM_EPIS_TEST = 50  # number of episodes for testing
ALPHA = 0.1  # learning rate for training

model = None
optimizer = None
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


def one_hot(i, n):
    if i >= n or i < 0:
        raise ValueError("Index out of bounds!")
    return torch.tensor([1 if j == i else 0 for j in range(n)])


def index2tuple(index, n):
    """Converts an index c to a tuple (a,b)"""
    return index // n, index % n

def epsilon_greedy(state_vector, epsilon, training=True):
    """Returns an action selected by an epsilon-greedy exploration policy

    Args:
        state_vector (torch.FloatTensor): extracted vector representation
        theta (np.ndarray): current weight matrix
        epsilon (float): the probability of choosing a random command

    Returns:
        (int, int): the indices describing the action/object to take
    """
    coin = np.random.random_sample()
    if coin < epsilon:
        action_index = np.random.randint(0,NUM_ACTIONS)
    else:
        if training:
            with torch.no_grad():
                q_values_action_next = policy(state_vector.to(torch.float32))
            action_index = q_values_action_next.argmax().item()
        else:
            with torch.no_grad():
                q_values_action_next = target(state_vector.to(torch.float32))
            action_index = q_values_action_next.argmax().item()


    return action_index

class DQN(nn.Module):
    """A simple deep Q network implementation.
    Computes Q values for each (action, object) tuple given an input state vector
    """

    def __init__(self, state_dim, action_dim, hidden = False, hidden_size=100):
        super(DQN, self).__init__()
        self.hidden = hidden
        if hidden: 
            self.state_encoder = nn.Linear(state_dim, hidden_size)
            self.state2action = nn.Linear(hidden_size, action_dim)
        else:
            self.state_encoder = nn.Linear(state_dim, action_dim)

    def forward(self, x):
        if self.hidden:
            state = F.relu(self.state_encoder(x))
            return self.state2action(state)
        else:
            return self.state_encoder(x)



# pragma: coderesponse template
def deep_q_learning():
    """Updates the weights of the DQN for a given transition

    Args:
        current_state_vector (torch.FloatTensor): vector representation of current state
        action_index (int): index of the current action
        object_index (int): index of the current object
        reward (float): the immediate reward the agent recieves from playing current command
        next_state_vector (torch.FloatTensor): vector representation of next state
        terminal (bool): True if this epsiode is over

    Returns:
        None
    """
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None]).view(-1,NUM_STATES).to(torch.float32)
    # print(batch.state)

    state_batch = torch.cat(batch.state).view(-1,NUM_STATES).to(torch.float32)
    # print(state_batch.shape)
    # print(batch.action)
    action_batch = torch.cat(batch.action).unsqueeze(-1)
    # print(action_batch.shape)
    reward_batch = torch.cat(batch.reward).unsqueeze(-1)

    state_action_values = policy(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE)
    with torch.no_grad():
        next_state_values[non_final_mask] = target(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
                         


    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy.parameters(), 100)
    optimizer.step()
# pragma: coderesponse end


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
    s = one_hot(s,NUM_STATES)
    terminal = False

    # print(current_room_desc,current_quest_desc)
    step_count = 0
    while not terminal and step_count < 100:
        # Choose next action and execute

        a = epsilon_greedy(
            s, epsilon, training=for_training)
        next_s, reward, terminal, _, _ = env.step(a)
        a = torch.tensor(a).unsqueeze(0)
        reward = torch.tensor([reward], dtype=torch.float32)
        if terminal:
            next_s = None
        else:
            next_s = one_hot(next_s, NUM_STATES)
        
        memory.push(s,a,next_s,reward)

        

        if for_training:
            # update Q-function.
            deep_q_learning()

            target_net_state_dict = target.state_dict()
            policy_net_state_dict = policy.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*ALPHA + target_net_state_dict[key]*(1-ALPHA)
            target.load_state_dict(target_net_state_dict)

        if not for_training:
            epi_reward = epi_reward + gamma_step * reward
            gamma_step = gamma_step * GAMMA
            

        # prepare next step
        s = next_s
        step_count += 1
    if not for_training:
        return epi_reward


def run_epoch():
    """Runs one epoch and returns reward averaged over test episodes"""
    rewards = []

    for _ in range(NUM_EPIS_TRAIN):
        run_episode(for_training=True)

    for _ in range(NUM_EPIS_TEST):
        rewards.append(run_episode(for_training=False))

    return np.mean(np.array(rewards))


def run():
    """Returns array of test reward per epoch for one run"""
    global target
    global policy
    global optimizer
    target = DQN(NUM_STATES, NUM_ACTIONS, hidden=True, hidden_size=16)
    policy  = DQN(NUM_STATES, NUM_ACTIONS, hidden=True, hidden_size=16)

    optimizer = optim.SGD(policy.parameters(), lr=ALPHA)

    single_run_epoch_rewards_test = []
    pbar = tqdm(range(NUM_EPOCHS), ncols=80)
    # return single_run_epoch_rewards_test, model
    for _ in pbar:
        single_run_epoch_rewards_test.append(run_epoch())
        pbar.set_description(
            "Avg reward: {:0.6f} | Ewma reward: {:0.6f}".format(
                np.mean(single_run_epoch_rewards_test),
                utils.ewma(single_run_epoch_rewards_test)))
    return single_run_epoch_rewards_test, target

if __name__ == '__main__':
    global env 
    env = gym.make('Taxi-v3')
    global NUM_ACTIONS
    global NUM_STATES
    NUM_ACTIONS = env.action_space.n
    NUM_STATES = env.observation_space.n
    epoch_rewards_test = []  # shape NUM_RUNS * NUM_EPOCHS
    models = []  # shape NUM_RUNS

    global BATCH_SIZE
    BATCH_SIZE = 32

    memory = ReplayMemory(1000)

    for _ in range(NUM_RUNS):
        rewards, model = run()
        epoch_rewards_test.append(rewards)
        models.append(model)

    i = np.argmax(epoch_rewards_test) // NUM_EPOCHS
    torch.save(models[i].state_dict(), "results/dqn_model_"+str(i)+".pt")

    epoch_rewards_test = np.array(epoch_rewards_test)

    x = np.arange(NUM_EPOCHS)
    fig, axis = plt.subplots()
    axis.plot(x, np.mean(epoch_rewards_test,
                         axis=0))  # plot reward per epoch averaged per run
    axis.set_xlabel('Epochs')
    axis.set_ylabel('reward')
    axis.set_title(('Linear: nRuns=%d, Epilon=%.2f, Epi=%d, alpha=%.4f' %
                    (NUM_RUNS, TRAINING_EP, NUM_EPIS_TRAIN, ALPHA)))
    plt.show()
    plt.savefig("runs/dqn_model_"+str(i)+".png")
