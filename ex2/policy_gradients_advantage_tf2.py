import gym
import numpy as np
import tensorflow as tf
import collections
from datetime import datetime
# import keras

tf.keras.backend.set_floatx('float64')

# ================================= TensorBoard settings ===============================================================

logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir)
file_writer.set_as_default()

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
    return loss_value, tape.gradient(loss_value, model.trainable_variabels)


# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def loss(model, x, y, _R_t, _estimated_value):
    y_ = model.predict(x)
    y = tf.reshape(tf.convert_to_tensor(y),shape=[1,2])
    return tf.keras.losses.categorical_crossentropy(y_true=y, y_pred=y_) * (_R_t - _estimated_value.item())


# =================================== Main Section =====================================================================


# Initialize the policy network
policy_net = PolicyNetwork(state_size)
value_net = ValueNetwork(state_size)


# policy_net.model.compile(loss=policy_loss(R_t, estimated_value),
# optimizer=keras.optimizers.Adam(learning_rate=policy_learning_rate))

policy_optimizer = tf.keras.optimizers.Adam(lr=policy_learning_rate)
value_net.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=value_learning_rate))

# Start training the agent with REINFORCE algorithm
solved = False
Transition = collections.namedtuple("Transition",
                                    ["state", "action", "reward", "next_state", "done", "estimated_value"])
episode_rewards = []
average_rewards = 0.0

for episode in range(max_episodes):
    state = env.reset()
    state = state.reshape([1, state_size])
    episode_transitions = []
    episode_reward = 0

    for step in range(max_steps):
        actions_distribution = tf.squeeze(policy_net.model.predict(state))
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
        episode_reward += reward

        if done:
            episode_rewards.append(episode_reward)
            average_rewards = np.mean(episode_rewards[episode-99:episode+1])
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

