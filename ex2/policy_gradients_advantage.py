import os
from datetime import datetime

import gym
import numpy as np
import tensorflow as tf
import collections
import pandas as pd
import matplotlib.pyplot as plt

# =========================================== Define hyperparameters ===================================================

render = False

env = gym.make('CartPole-v1')
np.random.seed(1)

state_size = 4
action_size = env.action_space.n

max_episodes = 5000
max_steps = 501
discount_factor = 0.99
learning_rate = 0.0004
value_learning_rate = 0.0004

experiment_name = 'policy_gradient_advantage'
results_dir = 'results/' + experiment_name + '/' + datetime.now().strftime("%Y%m%d-%H%M%S") + '/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


# ===================================== Models Definition ==============================================================


class PolicyNetwork:
    def __init__(self, _state_size, _action_size, _learning_rate, name='policy_network'):
        self.state_size = _state_size
        self.action_size = _action_size
        self.learning_rate = _learning_rate

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")
            self.estimated_value = tf.placeholder(tf.float32, name="estimated_value")

            self.W1 = tf.get_variable("W1", [self.state_size, 12],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b1 = tf.get_variable("b1", [12], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [12, self.action_size],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b2 = tf.get_variable("b2", [self.action_size], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * (self.R_t - self.estimated_value))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


class ValueNetwork:
    def __init__(self, _state_size, _learning_rate, name='value_network'):
        self.state_size = _state_size
        self.learning_rate = _learning_rate

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            self.W1 = tf.get_variable("W1", [self.state_size, 12],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b1 = tf.get_variable("b1", [12], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [12, 1], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b2 = tf.get_variable("b2", [1], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # output the value with no activation function applied to it
            self.estimated_value = tf.squeeze(self.output)

            self.loss = tf.squared_difference(self.estimated_value, self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


# ========================================== Util Methods ==============================================================

def plot_data(data_name, data, step):
    print("Ploting the {} data along the axis of {}".format(data_name, step))

    ax = pd.DataFrame(data).plot()
    ax.set_xlabel(step)
    ax.set_ylabel(data_name)
    ax.legend().remove()
    ax.set_title(experiment_name)
    plt.savefig(results_dir + '_' + data_name + '.png')

def plot_all_results():
    plot_data(data=all_episodes_rewards, data_name='rewards', step='episode')
    plot_data(data=avg_episodes_rewards, data_name='average_rewards', step='Last 100 episodes')
    plot_data(data=all_value_losses, data_name='value_loss', step='step')
    plot_data(data=all_policy_losses, data_name='policy_loss', step='step')


# ========================================== Main Method ===============================================================

tf.reset_default_graph()
policy = PolicyNetwork(state_size, action_size, learning_rate)
value = ValueNetwork(state_size, value_learning_rate)

# Start training the agent with REINFORCE algorithm
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    solved = False
    Transition = collections.namedtuple("Transition",
                                        ["state", "action", "reward", "next_state", "done", "estimated_value"])
    all_episodes_rewards = []
    avg_episodes_rewards = []
    all_value_losses = []
    all_policy_losses = []

    for episode in range(max_episodes):
        state = env.reset()
        state = state.reshape([1, state_size])
        episode_transitions = []
        episode_reward = 0

        for step in range(max_steps):
            actions_distribution, estimated_value = sess.run([policy.actions_distribution, value.estimated_value],
                                                             {policy.state: state, value.state: state})
            action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape([1, state_size])

            if render:
                env.render()

            action_one_hot = np.zeros(action_size)
            action_one_hot[action] = 1
            episode_transitions.append(
                Transition(state=state, action=action_one_hot, reward=reward, next_state=next_state, done=done,
                           estimated_value=estimated_value))
            episode_reward += reward

            if done:
                all_episodes_rewards.append(episode_reward)
                average_rewards = np.mean(all_episodes_rewards[episode - 99:episode + 1])
                avg_episodes_rewards.append(average_rewards)
                print("Episode {} Reward: {} Average over 100 episodes: {}".
                      format(episode, episode_reward, round(average_rewards, 2)))

                if average_rewards > 475:
                    print(' Solved at episode: ' + str(episode))
                    solved = True
                break
            state = next_state

        if solved:
            plot_all_results()
            break

        # Compute Rt for each time-step t and update the network's weights
        for t, transition in enumerate(episode_transitions):
            total_discounted_return = sum(
                discount_factor ** i * t.reward for i, t in enumerate(episode_transitions[t:]))  # Rt
            feed_dict = {policy.state: transition.state, policy.R_t: total_discounted_return,
                         policy.action: transition.action, policy.estimated_value: transition.estimated_value,
                         value.R_t: total_discounted_return, value.state: state}
            _, policy_loss, _, value_loss = sess.run([policy.optimizer, policy.loss, value.optimizer, value.loss],
                                                     feed_dict)
            all_policy_losses.append(policy_loss)
            all_value_losses.append(value_loss)

