"""Tabular QL agent"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils
import gymnasium as gym

DEBUG = False

GAMMA = 0.95  # discounted factor
TRAINING_EP = 0.5  # epsilon-greedy parameter for training
TESTING_EP = 0.05  # epsilon-greedy parameter for testing
NUM_RUNS = 10
NUM_EPOCHS = 200
NUM_EPIS_TRAIN = 25  # number of episodes for training at each epoch
NUM_EPIS_TEST = 50  # number of episodes for testing
ALPHA = 0.1  # learning rate for training

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
def tabular_q_learning(q_func, s, a,
                       reward, next_s,
                       terminal):
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
    if terminal:
        maxq_next_state = 0
    else:
        q_values_next_state = q_func[next_s, :]
        maxq_next_state = np.max(q_values_next_state)

    q_value = q_func[s, a]
    q_func[s,a] = (
        1 - ALPHA) * q_value + ALPHA * (reward + GAMMA * maxq_next_state)
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
    # print('here')   
    # print(current_room_desc,current_quest_desc)
    StepCount = 0
    while not terminal and StepCount < 1000:
        # Choose next action and execute

        a = epsilon_greedy(s, q_func, epsilon)
        next_s, reward, terminal, _, _ = env.step(a)
        # print(s,a,next_s,reward,terminal)
        
        if for_training:
            # update Q-function.
            tabular_q_learning(q_func, s ,a, reward, next_s, terminal)
            pass

        if not for_training:
            epi_reward = epi_reward + gamma_step * reward
            gamma_step = gamma_step * GAMMA
            

        # prepare next step
        s = next_s
        StepCount += 1
    if not for_training:
        return epi_reward


# pragma: coderesponse end


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
    global q_func
    q_func = np.zeros((NUM_STATES, NUM_ACTIONS))

    single_run_epoch_rewards_test = []
    pbar = tqdm(range(NUM_EPOCHS), ncols=80)
    for _ in pbar:
        single_run_epoch_rewards_test.append(run_epoch())
        pbar.set_description(
            "Avg reward: {:0.6f} | Ewma reward: {:0.6f}".format(
                np.mean(single_run_epoch_rewards_test),
                utils.ewma(single_run_epoch_rewards_test)))
    return single_run_epoch_rewards_test, q_func


if __name__ == '__main__':
    # Data loading and build the dictionaries that use unique index for each state
    # (dict_room_desc, dict_quest_desc) = framework.make_all_states_index()
    # NUM_ROOM_DESC = len(dict_room_desc)
    # NUM_QUESTS = len(dict_quest_desc)

    # set up the game
    # framework.load_game_data()
    global env 
    env = gym.make('Taxi-v3')
    NUM_ACTIONS = env.action_space.n
    NUM_STATES = env.observation_space.n
    

    epoch_rewards_test = []  # shape NUM_RUNS * NUM_EPOCHS
    q_functions = []
    for _ in range(NUM_RUNS):
        rewards, q_func = run()
        epoch_rewards_test.append(rewards)
        q_functions.append(q_func)

    i = np.argmax(epoch_rewards_test) // NUM_EPOCHS
    np.save("results/q_func"+str(i)+".npy", q_functions[i])
    epoch_rewards_test = np.array(epoch_rewards_test)

    x = np.arange(NUM_EPOCHS)
    fig, axis = plt.subplots()
    axis.plot(x, np.mean(epoch_rewards_test,
                         axis=0))  # plot reward per epoch averaged per run
    axis.set_xlabel('Epochs')
    axis.set_ylabel('reward')
    axis.set_title(('Tablular: nRuns=%d, Epilon=%.2f, Epi=%d, alpha=%.4f' %
                    (NUM_RUNS, TRAINING_EP, NUM_EPIS_TRAIN, ALPHA)))
    plt.savefig("runs/tabular.png")
