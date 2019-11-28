import random
from datetime import datetime

import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import collections

# ================================= TensorBoard settings ===============================================================

logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir)
file_writer.set_as_default()

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.config.experimental.list_physical_devices('GPU')
# ================================ Hyper-Parameters ====================================================================

episodes = 5000
state_size = 4

batch_size = 32
learning_rate = 0.001

discount_factor = 0.99
epsilon_greedy = 0.9
epsilon_greedy_decay_rate = 0.99
min_epsilon = 0.1

N = 2000
C = 16
# T = 500

replay_memory = collections.deque(maxlen=N)


# ======================================================================================================================


class Model3Layers:
    def __init__(self, _state_dim):
        self.model_3_layers = tf.keras.Sequential([
            keras.layers.Dense(24, input_dim=state_size, activation='relu'),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(2, activation='linear')
        ])


class Model5Layers:
    def __init__(self):
        self.model_3_layers = tf.keras.Sequential([
            keras.layers.Dense(1024, input_dim=state_size, activation='relu'),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(2, activation='linear')
        ])


# ======================================================================================================================


def choose_action_by_epsilon_greedy(_state):
    rand_num = np.random.uniform(0, 1)
    if rand_num <= epsilon_greedy:
        _step_to_take = env.action_space.sample()
    else:
        output = behavior_model.model_3_layers.predict(_state)
        _step_to_take = np.argmax(output)
    return _step_to_take


# Create the environment
env = gym.make('CartPole-v1')

target_model = Model3Layers(4)
behavior_model = Model3Layers(4)

target_model.model_3_layers.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
behavior_model.model_3_layers.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))


def remember(_state, _action, _reward, _next_state, _done):
    replay_memory.append((_state, _action, _reward, _next_state, _done))


def learn_from_memory():
    minibatch = random.sample(replay_memory, batch_size)
    states = np.zeros((batch_size, 4))
    targets = np.zeros((batch_size, 2))
    for index, (_state, _action, _reward, _next_state, _done) in enumerate(minibatch):
        target = _reward
        states[index] = _state
        if not _done:
            output = target_model.model_3_layers.predict(_next_state)
            target = _reward + discount_factor * np.argmax(output)

        target_tag = behavior_model.model_3_layers.predict(_state)
        target_tag[0][_action] = target
        targets[index] = target_tag[0]
    losses = behavior_model.model_3_layers.fit(x=states, y=targets, epochs=1, verbose=0)


for episode in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    time = 0

    while not done:
        time += 1
        action = choose_action_by_epsilon_greedy(state)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        remember(state, action, reward, next_state, done)
        state = next_state

        if done:
            print('Finished episode {} the score was {} epsilon is {}'.format(episode, time, epsilon_greedy))
            break

        if len(replay_memory) > batch_size:
            learn_from_memory()

        if epsilon_greedy > min_epsilon:
            epsilon_greedy = epsilon_greedy * epsilon_greedy_decay_rate

        if time % C == 0:
            # copy weights from one network to the other
            target_model.model_3_layers.set_weights(behavior_model.model_3_layers.get_weights())
