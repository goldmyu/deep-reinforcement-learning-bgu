import gym
import numpy as np

# ======================================================================================================================

train_episodes = 5000
max_steps_per_episode = 100

learning_rate = 0.1
discount_factor = 0.85
success_counter = 0

epsilon_greedy_rate = 0.9
epsilon_greedy_decay_rate = 0.999
min_epsilon = 0.1

reward_history = []

# Create the q-value lookup table in the size of states X actions and init it to zeros
q_value_lookup_table = np.zeros(shape=[16, 4])


# ======================================================================================================================

# Create the environment
env = gym.make('FrozenLake-v0')


def choose_action_by_epsilon_greedy(_state):
    rand_num = np.random.uniform(0, 1)
    if rand_num <= epsilon_greedy_rate:
        _step_to_take = env.action_space.sample()
    else:
        all_max_indexes = np.argwhere(q_value_lookup_table[state] == np.amax(q_value_lookup_table[state]))
        np.random.shuffle(all_max_indexes)
        _step_to_take = all_max_indexes.item(0)
    return _step_to_take


def update_q_table(_state, _action, _target):
    q_value_lookup_table[state, action] = (1 - learning_rate) * q_value_lookup_table[state, action] + \
                                         learning_rate * target


for episode in range(train_episodes):
    state = env.reset()
    env.render()
    epsilon_greedy_rate = max(min_epsilon, epsilon_greedy_rate * epsilon_greedy_decay_rate)

    for step in range(max_steps_per_episode):
        action = choose_action_by_epsilon_greedy(state)
        next_state, reward, done, info = env.step(action)  # take a random action
        if done:
            if next_state == 15:
                success_counter += 1
            target = reward
            update_q_table(state, action, target)
            break
        else:
            target = reward + discount_factor * np.argmax(q_value_lookup_table[next_state])
            update_q_table(state, action, target)
        env.render()
        state = next_state

print("number of success is {}".format(success_counter))
print("Q-value table is {}".format(q_value_lookup_table))
env.close()
