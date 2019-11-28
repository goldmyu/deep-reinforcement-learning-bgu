from datetime import datetime

import gym
import numpy as np
import tensorflow as tf

# ======================================================================================================================

train_episodes = 5000
max_steps_per_episode = 100

learning_rate = 0.1
discount_factor = 0.99
success_counter = 0

epsilon_greedy_rate = 0.9
epsilon_greedy_decay_rate = 0.99
min_epsilon = 0.1

steps_to_goal_history = []
reward_history = []

# Create the q-value lookup table in the size of states X actions and init it to zeros
q_value_lookup_table = np.zeros(shape=[16, 4])

# ================================= TensorBoard settings ===============================================================

logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir)
file_writer.set_as_default()

# ======================================================================================================================


# Create the environment
env = gym.make('FrozenLake-v0')


def choose_action_by_epsilon_greedy(_state):
    rand_num = np.random.uniform(0, 1)
    if rand_num <= epsilon_greedy_rate:
        _step_to_take = env.action_space.sample()
    else:
        all_max_indexes = np.argwhere(q_value_lookup_table[_state] == np.amax(q_value_lookup_table[_state]))
        np.random.shuffle(all_max_indexes)
        _step_to_take = all_max_indexes.item(0)
    return _step_to_take


def moving_average(rewards_list, _episode, n=100):
    if _episode == 0:
        tf.summary.scalar('reward_moving_avg', data=0, step=_episode)
    else:
        avg = np.sum(rewards_list[_episode-n:_episode]) / n
        tf.summary.scalar('reward_moving_avg', data=avg, step=_episode)


def update_q_table(_state, _action, _target):
    q_value_lookup_table[_state, _action] = (1 - learning_rate) * q_value_lookup_table[_state, _action] + \
                                            (learning_rate * _target)


for episode in range(train_episodes):
    state = env.reset()
    env.render()
    epsilon_greedy_rate = max(min_epsilon, epsilon_greedy_rate * epsilon_greedy_decay_rate)

    if episode % 100 == 0 and episode != 0:
        steps_to_goal_avg = np.average(steps_to_goal_history[episode - 100:episode - 1])
        tf.summary.scalar('avg_steps_to_goal', data=steps_to_goal_avg, step=episode)

    for step in range(max_steps_per_episode):
        action = choose_action_by_epsilon_greedy(state)
        next_state, reward, done, info = env.step(action)  # take a random action

        if done:
            if next_state == 15:
                success_counter += 1
                tf.summary.scalar('steps_to_goal', data=step, step=episode)
                steps_to_goal_history.append(step)
            else:
                tf.summary.scalar('steps_to_goal', data=100, step=episode)
                steps_to_goal_history.append(100)

            reward_history.append(reward)
            tf.summary.scalar('reward', data=reward, step=episode)
            target = reward
            update_q_table(state, action, target)
            env.render()
            break
        else:
            max_action = np.argmax(q_value_lookup_table[next_state])
            target = reward + discount_factor * q_value_lookup_table[next_state, max_action]
            update_q_table(state, action, target)
        state = next_state
        env.render()
    moving_average(reward_history, episode)

print("number of success is {}".format(success_counter))
print("Q-value table is {}".format(q_value_lookup_table))
env.close()
