import os
import random
from datetime import datetime

import gym
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential

import collections

# ================================= TensorBoard settings ===============================================================

logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir)
file_writer.set_as_default()
#

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# ================================ Hyper-Parameters ====================================================================

episodes = 5000
state_size = 4

batch_size = 64
learning_rate = 0.001

discount_factor = 1
epsilon_greedy = 1
epsilon_greedy_decay_rate = 0.999
min_epsilon = 0.01

reward_history = []
first_time_avg_475 = True
global_step = 0

N = 2000
C = 64

replay_memory = collections.deque(maxlen=N)


# =================================== Network definition ===============================================================


class Model3Layers:
    def __init__(self, _state_dim):
        self.model = Sequential([
            keras.layers.Dense(24, input_dim=state_size, activation='relu'),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(2, activation='linear')
        ])


class Model5Layers:
    def __init__(self, _state_dim):
        self.model = keras.Sequential([
            keras.layers.Dense(48, input_dim=state_size, activation='relu'),
            keras.layers.Dense(48, activation='relu'),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(2, activation='linear')
        ])


# =================================== Util methods =====================================================================


def choose_action_by_epsilon_greedy(_state):
    rand_num = np.random.uniform(0, 1)
    if rand_num <= epsilon_greedy:
        _step_to_take = env.action_space.sample()
    else:
        output = behavior_model.model.predict(_state)
        _step_to_take = np.argmax(output)
    return _step_to_take


def remember(_state, _action, _reward, _next_state, _done):
    replay_memory.append((_state, _action, _reward, _next_state, _done))


def learn_from_memory(_step,_global_step):
    minibatch = random.sample(replay_memory, batch_size)
    states = np.zeros((batch_size, 4))
    states_in_minibatch = np.asarray([x[0][0] for x in minibatch])
    targets = behavior_model.model.predict(states_in_minibatch)
    next_states = np.asanyarray([x[3][0] for x in minibatch])
    outputs = target_model.model.predict(next_states)

    for index, (_state, _action, _reward, _next_state, _done) in enumerate(minibatch):
        target = _reward
        states[index] = _state
        if not _done:
            target = _reward + discount_factor * np.max(outputs[index])

        targets[index, _action] = target
    loss = behavior_model.model.fit(x=states, y=targets, epochs=1, verbose=0)
    tf.summary.scalar('loss', data=loss.history["loss"][0], step=_global_step)


def check_reward_avg(rewards_list, _episode, _first_time_avg_475,n=100):
    if _episode > 0:
        avg = np.sum(rewards_list[_episode-n:_episode]) / n
        tf.summary.scalar('reward_moving_avg', data=avg, step=_episode)
        if avg > 475 and _first_time_avg_475:
            _first_time_avg_475 = False
            print('Reward avg is above 475 for 100 last episodes, and the episode is {}'.format(episode))
    return _first_time_avg_475

# =================================== Main Section =====================================================================


# Create the environment
env = gym.make('CartPole-v1')

target_model = Model5Layers(4)
behavior_model = Model5Layers(4)

target_model.model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
behavior_model.model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=learning_rate))


for episode in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    step = 0

    first_time_avg_475 = check_reward_avg(reward_history, episode, first_time_avg_475)

    while not done:
        step += 1
        global_step += 1
        action = choose_action_by_epsilon_greedy(state)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        remember(state, action, reward, next_state, done)
        state = next_state

        if done:
            reward_history.append(step)
            print('Finished episode {} the score was {} epsilon is {}'.format(episode, step, epsilon_greedy))
            tf.summary.scalar('reward', data=step, step=episode)
            break

        if len(replay_memory) > batch_size:
            learn_from_memory(step,global_step)

        if epsilon_greedy > min_epsilon:
            epsilon_greedy = epsilon_greedy * epsilon_greedy_decay_rate

        if global_step % C == 0:
            # copy weights from one network to the other
            target_model.model.set_weights(behavior_model.model.get_weights())
