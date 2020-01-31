import time
from datetime import datetime
import os

import gym
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# =========================================== Define hyperparameters ===================================================

env = gym.make('CartPole-v1')
np.random.seed(1)

state_size = 6
action_size = 3
pad_state_size_cart_pole = [0, 0]
valid_action_space_cart_pole = 2

max_episodes = 1000
max_steps = 501
discount_factor = 0.99
policy_learning_rate = 0.0007
value_learning_rate = 0.006

experiment_name = 'acrobot_to_cartpole_model'
results_dir = 'results/' + experiment_name + '/' + datetime.now().strftime("%Y%m%d-%H%M%S") + '/'
acrobot_model_dir = 'results/acrobot_model/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

start_time = time.time()

# ===================================== Models Definition ==============================================================

class PolicyNetwork:
    def __init__(self, _state_size, _action_size, _learning_rate, name='policy'):
        self.state_size = _state_size
        self.action_size = _action_size
        self.learning_rate = _learning_rate

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")
            self.estimated_value = tf.placeholder(tf.float32, name="estimated_value")
            with tf.variable_scope('policy_network'):
                self.W1 = tf.get_variable("W1", [self.state_size, 12],
                                          initializer=tf.contrib.layers.xavier_initializer(seed=0))
                self.b1 = tf.get_variable("b1", [12], initializer=tf.zeros_initializer())

                self.W2 = tf.get_variable("W2", [12, 12],
                                          initializer=tf.contrib.layers.xavier_initializer(seed=0))
                self.b2 = tf.get_variable("b2", [12], initializer=tf.zeros_initializer())

            self.W3 = tf.get_variable("W3", [12, self.action_size],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b3 = tf.get_variable("b3", [self.action_size], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            # self.A1 = tf.nn.relu(self.Z1)
            self.A1 = tf.nn.sigmoid(self.Z1)
            self.Z2 = tf.add(tf.matmul(self.A1, self.W2), self.b2)
            self.A2 = tf.nn.sigmoid(self.Z2)
            self.output = tf.add(tf.matmul(self.A2, self.W3), self.b3)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


class ValueNetwork:
    def __init__(self, _state_size, _learning_rate, name='value_network'):
        self.state_size = _state_size
        self.learning_rate = _learning_rate

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            self.W1 = tf.get_variable("W1", [self.state_size, 10],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b1 = tf.get_variable("b1", [10], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [10, 1], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b2 = tf.get_variable("b2", [1], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            # self.A1 = tf.nn.relu(self.Z1)
            self.A1 = tf.nn.sigmoid(self.Z1)
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


def plot_all_results(all_episodes_rewards, avg_episodes_rewards, loss_actor, loss_critic):
    plot_data(data=all_episodes_rewards, data_name='rewards', step='episode')
    plot_data(data=avg_episodes_rewards, data_name='average_rewards', step='Last 100 episodes')
    plot_data(data=loss_actor, data_name='actor_loss', step='step')
    plot_data(data=loss_critic, data_name='critic_loss', step='step')


# ========================================== Main Method ===============================================================


def train(policy, value):
    # Start training the agent with REINFORCE algorithm
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver({'policy_network/W1': policy.W1, 'policy_network/b1': policy.b1,
                                'policy_network/W2': policy.W2, 'policy_network/b2': policy.b2})
        saver.restore(sess, acrobot_model_dir + "model.ckpt")

        all_episodes_rewards = []
        avg_episodes_rewards = []
        loss_critic = []
        loss_actor = []

        for episode in range(max_episodes):
            state = env.reset()
            state = np.append(state, [0, 0])
            state = state.reshape([1, state_size])
            episode_reward = 0

            for step in range(max_steps):
                actions_distribution, value_state = sess.run([policy.actions_distribution, value.estimated_value],
                                                             {policy.state: state, value.state: state})

                actions_distribution[2] = 0
                action_sum = actions_distribution[0] + actions_distribution[1]
                actions_distribution[0] = actions_distribution[0] / action_sum
                actions_distribution[1] = actions_distribution[1] / action_sum

                action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
                next_state, reward, done, _ = env.step(action)
                next_state = np.append(next_state, [0, 0])
                next_state = next_state.reshape([1, state_size])

                action_one_hot = np.zeros(action_size)
                action_one_hot[action] = 1
                episode_reward += reward

                value_next_state = sess.run(value.estimated_value, {value.state: next_state})
                td_target = reward
                if not done:
                    td_target += (discount_factor * value_next_state)
                lambda_value = td_target - value_state

                feed_dict_value = {value.state: state, value.R_t: td_target}
                _, loss_value = sess.run([value.optimizer, value.loss], feed_dict_value)
                loss_critic.append(loss_value)
                feed_dict_policy = {policy.state: state, policy.R_t: lambda_value, policy.action: action_one_hot}
                _, loss_policy = sess.run([policy.optimizer, policy.loss], feed_dict_policy)
                loss_actor.append(loss_policy)

                if done:
                    all_episodes_rewards.append(episode_reward)
                    average_rewards = np.mean(all_episodes_rewards[episode - 99:episode + 1])
                    avg_episodes_rewards.append(average_rewards)
                    print("Episode {} Reward: {} Average over 100 episodes: {}".
                          format(episode, episode_reward, round(average_rewards, 2)))

                    if average_rewards > 475:
                        print(' Solved at episode: ' + str(episode))
                        saver.save(sess, results_dir)
                        plot_all_results(all_episodes_rewards, avg_episodes_rewards, loss_actor, loss_critic)
                        return True
                    if episode > 100 and average_rewards < 20:
                        return False
                    if episode > 700 and average_rewards < 350:
                        return False
                    break
                state = next_state
        return False


def main():
    goal_reached = False

    while not goal_reached:
        tf.reset_default_graph()

        policy = PolicyNetwork(state_size, action_size, policy_learning_rate)
        value = ValueNetwork(state_size, value_learning_rate)
        # saver = tf.train.Saver()

        goal_reached = train(policy, value)

        end_time = time.time()
        print('Run time was {}'.format(end_time - start_time))


if __name__ == "__main__":
    main()
