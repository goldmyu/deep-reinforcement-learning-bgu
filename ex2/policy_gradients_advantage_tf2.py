import gym
import numpy as np
import tensorflow as tf
import collections
from datetime import datetime

# import keras
# from keras.models import Sequential
tf.keras.backend.set_floatx('float64')

# ================================= TensorBoard settings ===============================================================

logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
train_summary_writer = tf.summary.create_file_writer(logdir)
train_summary_writer.set_as_default()

# ================================ Hyper-Parameters ====================================================================

env = gym.make('CartPole-v1')
np.random.seed(1)

state_size = 4
action_size = env.action_space.n

max_episodes = 5000
max_steps = 501
discount_factor = 0.99
policy_learning_rate = 0.0004
value_learning_rate = 0.0004

render = False


# =================================== Network definition ===============================================================
#
# class PolicyNetwork:
#     def __init__(self, _state_size, _action_size, _learning_rate, name='policy_network'):
#         self.state_size = _state_size
#         self.action_size = _action_size
#         self.learning_rate = _learning_rate
#
#         with tf.variable_scope(name):
#
#             self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
#             self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
#             self.R_t = tf.placeholder(tf.float32, name="total_rewards")
#             self.estimated_value = tf.placeholder(tf.float32, name="estimated_value")
#
#             self.W1 = tf.get_variable("W1", [self.state_size, 12], initializer=tf.contrib.layers.xavier_initializer(seed=0))
#             self.b1 = tf.get_variable("b1", [12], initializer=tf.zeros_initializer())
#             self.W2 = tf.get_variable("W2", [12, self.action_size], initializer=tf.contrib.layers.xavier_initializer(seed=0))
#             self.b2 = tf.get_variable("b2", [self.action_size], initializer=tf.zeros_initializer())
#
#             self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
#             self.A1 = tf.nn.relu(self.Z1)
#             self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)
#
#             # Softmax probability distribution over actions
#             self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
#             # Loss with negative log probability
#             self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
#             self.loss = tf.reduce_mean(self.neg_log_prob * (self.R_t-self.estimated_value))
#             self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

#
# class ValueNetwork:
#     def __init__(self, _state_size, _learning_rate, name='value_network'):
#         self.state_size = _state_size
#         self.learning_rate = _learning_rate
#
#         with tf.variable_scope(name):
#
#             self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
#             self.R_t = tf.placeholder(tf.float32, name="total_rewards")
#
#             self.W1 = tf.get_variable("W1", [self.state_size, 12], initializer=tf.contrib.layers.xavier_initializer(seed=0))
#             self.b1 = tf.get_variable("b1", [12], initializer=tf.zeros_initializer())
#             self.W2 = tf.get_variable("W2", [12,1], initializer=tf.contrib.layers.xavier_initializer(seed=0))
#             self.b2 = tf.get_variable("b2", [1], initializer=tf.zeros_initializer())
#
#             self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
#             self.A1 = tf.nn.relu(self.Z1)
#             self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)
#
#             # output the value with no activation function applied to it
#             self.estimated_value = tf.squeeze(self.output)
#
#             self.loss = tf.squared_difference(self.estimated_value, self.R_t)
#             self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


class PolicyNetwork:
    def __init__(self, _state_size):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(12, input_dim=_state_size, activation='relu'),
            # keras.layers.Dense(12, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax'),
        ])


class ValueNetwork:
    def __init__(self, _state_size):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(12, input_dim=_state_size, activation='relu'),
            tf.keras.layers.Dense(1)
        ])


# =================================== Util methods =====================================================================


def grad(model, inputs, targets, _R_t, _estimated_value):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets,_R_t, _estimated_value)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def loss(model, x, y, _R_t, _estimated_value):
    y_ = model(x)
    y = tf.reshape(tf.convert_to_tensor(y),shape=[1,2])
    return tf.keras.losses.categorical_crossentropy(y_true=y, y_pred=y_) * (_R_t - _estimated_value.item())


# =================================== Main Section =====================================================================


# Initialize the policy network
policy_net = PolicyNetwork(state_size)
value_net = ValueNetwork(state_size)


policy_optimizer = tf.keras.optimizers.Adam(learning_rate=policy_learning_rate)
# policy_net.model.compile(loss=policy_loss(R_t, estimated_value),optimizer=keras.optimizers.Adam(learning_rate=policy_learning_rate))
value_net.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=value_learning_rate))

# Start training the agent with REINFORCE algorithm
solved = False
Transition = collections.namedtuple("Transition",
                                    ["state", "action", "reward", "next_state", "done", "estimated_value"])
episode_rewards = np.zeros(max_episodes)
average_rewards = 0.0

for episode in range(max_episodes):
    state = env.reset()
    state = state.reshape([1, state_size])
    episode_transitions = []

    for step in range(max_steps):
        # actions_distribution, estimated_value = sess.run([policy_net.actions_distribution, value_net.estimated_value], {policy_net.state: state, value_net.state: state})
        actions_distribution = tf.squeeze(policy_net.model(state))
        estimated_value = value_net.model.predict(state)
        action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
        next_state, reward, done, _ = env.step(action)
        next_state = next_state.reshape([1, state_size])

        if render:
            env.render()

        action_one_hot = np.zeros(action_size)
        action_one_hot[action] = 1
        episode_transitions.append(Transition(state=state, action=action_one_hot, reward=reward,
                                              next_state=next_state, done=done, estimated_value=estimated_value))
        episode_rewards[episode] += reward

        if done:
            if episode > 98:
                # Check if solved
                average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
            print("Episode {} Reward: {} Average over 100 episodes: {}".
                  format(episode, episode_rewards[episode], round(average_rewards, 2)))

            tf.summary.scalar(name='reward_moving_avg', data=round(average_rewards, 2),step=episode)
            tf.summary.scalar(name='reward', data=reward, step=episode)

            if average_rewards > 475:
                print(' Solved at episode: ' + str(episode))
                solved = True
            break
        state = next_state

    if solved:
        break

    # Compute Rt for each time-step t and update the network's weights
    for t, transition in enumerate(episode_transitions):
        total_discounted_return = sum(
            discount_factor ** i * t.reward for i, t in enumerate(episode_transitions[t:]))  # Rt

        loss_value, grads = grad(policy_net.model, transition.state, transition.action,
                                 _R_t=total_discounted_return, _estimated_value=transition.estimated_value)
        policy_optimizer.apply_gradients(zip(grads, policy_net.model.trainable_variables))

        total_discounted_return = tf.reshape(tf.convert_to_tensor(total_discounted_return),shape=[1])
        value_net.model.fit(x=transition.state, y=total_discounted_return, epochs=1, verbose=0)
