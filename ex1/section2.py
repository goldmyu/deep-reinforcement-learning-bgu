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

# ================================ Hyper-Parameters ====================================================================

batch_size = 32
learning_rate = 0.1

epsilon_greedy = 0.9
epsilon_greedy_decay_rate = 0.99
min_epsilon = 0.1

de = collections.deque(maxlen=2000)


C = 500
M = 5000
T = 500

# ======================================================================================================================


# Create the environment
env = gym.make('CartPole-v1')

model_3_layers = tf.keras.Sequential([
    keras.layers.Flatten(input_shape=(2,)),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(2, activation='linear')
])

model_5_layers = tf.keras.Sequential([
    keras.layers.Flatten(input_shape=(2,)),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(2, activation='linear')
])


def choose_action_by_epsilon_greedy(_state):
    pass


for episode in range(M):
    state = env.reset()
    for step in range(T):
        action = choose_action_by_epsilon_greedy(state)
    