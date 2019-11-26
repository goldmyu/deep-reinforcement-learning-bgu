import gym
import numpy as np
import tensorflow as tf

# ======================================================================================================================

train_episodes = 5000
max_steps_per_episode = 100
learning_rate = 0.1
discount_factor = 0.99
epsilon_greedy_rate = 0.2
epsilon_greedy_decay_rate = 0.005
reward_history = []

# ======================================================================================================================

# Create the environment
env = gym.make('FrozenLake-v0')
observation = env.reset()

# Create the q-value lookup table in the size of states X actions and init it to zeros
q_value_lookup_table = np.zeros(shape=[16, 4])


def choose_step_by_epsilon_greedy(state):
    rand_num = np.random.uniform(0,1)
    if rand_num <= epsilon_greedy_rate:
        _step_to_take = env.action_space.sample()
    else:
        _step_to_take = np.argmax(q_value_lookup_table[state])
    return _step_to_take


def update_q_table(state, next_state, action, _reward):
    q_value_lookup_table[state,action] = q_value_lookup_table[state,action] + \
                                         learning_rate*(_reward +
                                                        discount_factor * np.argmax(q_value_lookup_table[next_state]) -
                                                        q_value_lookup_table[state,action])


for episode in range(train_episodes):
    for step in range(max_steps_per_episode):
        env.render()
        step_to_take = choose_step_by_epsilon_greedy(observation)
        observation, reward, done, info = env.step(step_to_take)  # take a random action

        if done:

        update_q_table()



env.close()