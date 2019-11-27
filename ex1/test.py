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
# Create the q-value lookup table in the size of states X actions and init it to zerosv
q_value_lookup_table = np.zeros(shape=[16, 4])

reward_history = []

# ======================================================================================================================

# Create the environment
env = gym.make('FrozenLake-v0')



def choose_action_by_epsilon_greedy(state):
    rand_num = np.random.uniform(0,1)
    if rand_num <= epsilon_greedy_rate:
        _step_to_take = env.action_space.sample()
    else:
        _step_to_take = np.argmax(q_value_lookup_table[state])
    return _step_to_take




def update_q_table(state, action, target):
    q_value_lookup_table[state,action] = (1 - learning_rate) * q_value_lookup_table[state,action] + \
                                         learning_rate * target
for episode in range(train_episodes):
    state = env.reset()
    for step in range(max_steps_per_episode):
        env.render()
        action = choose_action_by_epsilon_greedy(state)
        next_state, reward, done, info = env.step(action)  # take a random action
        if done:
            target = reward
            update_q_table(state, action, target)
            break
        else:
            target = reward + discount_factor * np.argmax(q_value_lookup_table[next_state])
            update_q_table(state, action, target)
        state = next_state



env.close()