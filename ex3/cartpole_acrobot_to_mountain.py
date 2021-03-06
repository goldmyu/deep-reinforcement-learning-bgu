import time
from datetime import datetime
import os

import gym
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# =========================================== Define hyperparameters ===================================================

env = gym.make('MountainCarContinuous-v0')
np.random.seed(1)

state_size = 6
action_size = 3
pad_state_size_cart_pole = [0, 0, 0, 0]
valid_action_space_cart_pole = 2

max_episodes = 1000
max_steps = 10000
discount_factor = 0.99
policy_learning_rate = 0.0002
value_learning_rate = 0.001

experiment_name = 'cartpole_acrobot_to_mountain'
results_dir = 'results/' + experiment_name + '/' + datetime.now().strftime("%Y%m%d-%H%M%S") + '/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

start_time = time.time()

# ===================================== Models Definition ==============================================================

class PolicyNetwork:
    def __init__(self, _state_size, _action_size, _learning_rate, name='policy_network'):
        self.state_size = _state_size
        self.action_size = _action_size
        self.learning_rate = _learning_rate

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, 1, name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")
            self.estimated_value = tf.placeholder(tf.float32, name="estimated_value")
            self.acrobot_1 = tf.placeholder(tf.float32, [None, 12], name="state")
            self.cartpole_1 = tf.placeholder(tf.float32, [None, 12], name="state")

            self.W1 = tf.get_variable("W1", [self.state_size, 12],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b1 = tf.get_variable("b1", [12], initializer=tf.zeros_initializer())

            self.W2 = tf.get_variable("W2", [12, self.action_size],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b2 = tf.get_variable("b2", [self.action_size], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.Z1 = tf.add(self.Z1, self.acrobot_1)
            self.Z1 = tf.add(self.Z1, self.cartpole_1)

            # self.A1 = tf.nn.relu(self.Z1)
            self.A1 = tf.nn.sigmoid(self.Z1)
            self.Z2 = tf.add(tf.matmul(self.A1, self.W2), self.b2)
            self.output = tf.nn.elu(self.Z2)


            self.mean = tf.layers.dense(self.output, 1, None, tf.contrib.layers.xavier_initializer(seed=0))
            self.var = tf.layers.dense(self.output, 1, None, tf.contrib.layers.xavier_initializer(seed=0))

            self.var = tf.nn.softplus(self.var) + 1e-5

            self.action_dist = tf.contrib.distributions.Normal(self.mean, self.var)
            self.action = self.action_dist.sample(1)
            self.action = tf.clip_by_value(self.action, clip_value_min=-1, clip_value_max=1)

            loss = -tf.log(self.action_dist.prob(self.action) + 1e-5) * self.R_t
            self.loss = loss - (self.action_dist.entropy() * 1e-1)
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


class CartPolePolicy:
    def __init__(self, _state_size, _action_size, _learning_rate, name='cartpole_policy'):
        self.state_size = _state_size
        self.action_size = _action_size
        self.learning_rate = _learning_rate

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")
            self.estimated_value = tf.placeholder(tf.float32, name="estimated_value")

            self.W1 = tf.get_variable("W1", [self.state_size, 12],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0), trainable=False)
            self.b1 = tf.get_variable("b1", [12], initializer=tf.zeros_initializer(), trainable=False)

            self.W2 = tf.get_variable("W2", [12, self.action_size],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0), trainable=False)
            self.b2 = tf.get_variable("b2", [self.action_size], initializer=tf.zeros_initializer(), trainable=False)

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            # self.A1 = tf.nn.relu(self.Z1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)


class AcrobotPolicy:
    def __init__(self, _state_size, _action_size, _learning_rate, name='acrobot_policy'):
        self.state_size = _state_size
        self.action_size = _action_size
        self.learning_rate = _learning_rate

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")
            self.estimated_value = tf.placeholder(tf.float32, name="estimated_value")

            self.W1 = tf.get_variable("W1", [self.state_size, 12],initializer=tf.contrib.layers.xavier_initializer(seed=0), trainable=False)
            self.b1 = tf.get_variable("b1", [12], initializer=tf.zeros_initializer(), trainable=False)
            self.W2 = tf.get_variable("W2", [12, 12], initializer=tf.contrib.layers.xavier_initializer(seed=0), trainable=False)
            self.b2 = tf.get_variable("b2", [12], initializer=tf.zeros_initializer(), trainable=False)
            self.W3 = tf.get_variable("W3", [12, self.action_size],initializer=tf.contrib.layers.xavier_initializer(seed=0), trainable=False)
            self.b3 = tf.get_variable("b3", [self.action_size], initializer=tf.zeros_initializer(), trainable=False)

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.Z2 = tf.add(tf.matmul(self.A1, self.W2), self.b2)
            self.A2 = tf.nn.relu(self.Z2)
            self.output = tf.add(tf.matmul(self.A2, self.W3), self.b3)


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


def train(policy, value,cartpole_policy,acrobot_policy, scaler):
    # Start training the agent with REINFORCE algorithm
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        cartpole_res = tf.train.Saver({'policy_network/W1': cartpole_policy.W1, 'policy_network/W2': cartpole_policy.W2,
                                      'policy_network/b1': cartpole_policy.b1, 'policy_network/b2': cartpole_policy.b2})
        cartpole_res.restore(sess, 'results/cartpole_model/'+"model.ckpt")

        acrobot_res = tf.train.Saver({'policy_network/W1': acrobot_policy.W1, 'policy_network/W2': acrobot_policy.W2,'policy_network/W3': acrobot_policy.W3,
                                      'policy_network/b1': acrobot_policy.b1, 'policy_network/b2': acrobot_policy.b2, 'policy_network/b3': acrobot_policy.b3})
        acrobot_res.restore(sess, 'results/acrobot_model/'+"model.ckpt")
        all_episodes_rewards = []
        avg_episodes_rewards = []
        loss_critic = []
        loss_actor = []

        for episode in range(max_episodes):
            state = env.reset()
            state = scaler.transform([state])[0]
            state = np.append(state, [0, 0, 0, 0])
            state = state.reshape([1, state_size])
            episode_reward = 0

            for step in range(max_steps):
                cartpoleA1 = sess.run(cartpole_policy.A1, {cartpole_policy.state:state})
                acrobatA2 = sess.run(acrobot_policy.A1, {acrobot_policy.state:state})
                action, value_state = sess.run([policy.action, value.estimated_value],
                                                             {policy.state: state, value.state: state, policy.acrobot_1: acrobatA2, policy.cartpole_1: cartpoleA1})

                next_state, reward, done, _ = env.step(action)
                next_state = np.squeeze(next_state)
                next_state = (scaler.transform([next_state]))[0]
                next_state = np.append(next_state, [0, 0, 0, 0])
                next_state = next_state.reshape([1, state_size])

                episode_reward += reward

                value_next_state = sess.run(value.estimated_value, {value.state: next_state})
                td_target = reward
                if not done:
                    td_target += (discount_factor * value_next_state)
                lambda_value = td_target - value_state

                feed_dict_value = {value.state: state, value.R_t: td_target}
                _, loss_value = sess.run([value.optimizer, value.loss], feed_dict_value)
                loss_critic.append(loss_value)

                feed_dict_policy = {policy.state: state, policy.R_t: lambda_value, policy.action: action, policy.acrobot_1:acrobatA2, policy.cartpole_1:cartpoleA1}
                _, loss_policy = sess.run([policy.optimizer, policy.loss], feed_dict_policy)
                loss_actor.append(loss_policy)
                if reward >80:
                    print("Episode {} Reward: {} step: {}".
                          format(episode, episode_reward, step))
                if done:
                    all_episodes_rewards.append(episode_reward)
                    average_rewards = np.mean(all_episodes_rewards[episode - 99:episode + 1])
                    avg_episodes_rewards.append(average_rewards)
                    print("Episode {} Reward: {} Average over 100 episodes: {}".
                          format(episode, episode_reward, round(average_rewards, 2)))

                    if episode > 80 and average_rewards > 80:
                        print(' Solved at episode: ' + str(episode))
                        saver = tf.train.Saver()

                        saver.save(sess, results_dir+"model.ckpt")
                        plot_all_results(all_episodes_rewards, avg_episodes_rewards, loss_actor, loss_critic)
                        return True
                    if episode >= 0 and average_rewards < -20:
                        return False
                    break
                state = next_state
        return False


def main():
    goal_reached = False

    state_space_samples = np.array([env.observation_space.sample() for x in range(10000)])
    scaler = StandardScaler()
    scaler.fit(state_space_samples)

    while not goal_reached:
        tf.reset_default_graph()
        policy = PolicyNetwork(state_size, action_size, policy_learning_rate)
        value = ValueNetwork(state_size, value_learning_rate)
        cartpole_policy = CartPolePolicy(state_size, action_size, policy_learning_rate)
        acrobot_policy = AcrobotPolicy(state_size, action_size, policy_learning_rate)

        goal_reached = train(policy, value,cartpole_policy,acrobot_policy, scaler)

        end_time = time.time()
        print('Run time was {}'.format(end_time - start_time))


if __name__ == "__main__":
    main()
