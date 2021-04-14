"""
DQN agent
"""

import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Flatten


class DqnAgent:
    """
    DQN agent with production policy and benchmark
    """

    # pylint: disable=too-many-arguments
    def __init__(self, state_space, action_space, gamma, lr, verbose,
                 checkpoint_location, model_location, persist_progress_option, mode, epsilon, train_period, perception):
        self.action_space = action_space
        self.mode = mode
        self.state_space = state_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.persist_progress_option = persist_progress_option
        self.verbose = verbose
        self.model_location = model_location
        self.checkpoint_location = checkpoint_location
        self.train_period = train_period
        self.perception = perception
        if self.verbose == 'init':
            print('Construct DQN agent with: ')
            print('Action space: ')
            print(action_space)
            print('State space: ')
            print(state_space)
        if self.perception:
            self.q_net = self._build_perception_dqn_model(state_space=state_space,
                                                          action_space=action_space, learning_rate=lr)
            self.target_q_net = self._build_perception_dqn_model(state_space=state_space,
                                                                 action_space=action_space, learning_rate=lr)
        else:
            self.q_net = self._build_dqn_model(state_space=state_space,
                                               action_space=action_space, learning_rate=lr)
            self.target_q_net = self._build_dqn_model(state_space=state_space,
                                                      action_space=action_space, learning_rate=lr)

        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                              net=self.q_net)
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, self.checkpoint_location, max_to_keep=10)
        # if self.persist_progress_option == 'all':
        #     if self.mode == 'train':
        #         self.load_checkpoint()
        #         self.update_target_network()
        #     if self.mode == 'test':
        #         self.load_model()

    @staticmethod
    def _build_dqn_model(state_space, action_space, learning_rate):
        """
        Builds a neural network for the agent

        :param state_space: state specification
        :param action_space: action specification
        :param learning_rate: learning rate
        :return: model
        """
        q_net = Sequential()
        q_net.add(Dense(128, input_dim=state_space, activation='relu',
                        kernel_initializer='he_uniform'))
        q_net.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        q_net.add(Dense(action_space, activation='linear',
                        kernel_initializer='he_uniform'))
        q_net.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                      loss='mse')
        q_net.summary()
        return q_net

    @staticmethod
    def _build_perception_dqn_model(state_space, action_space, learning_rate):
        """
        Builds a neural network for the agent

        :param state_space: state specification
        :param action_space: action specification
        :param learning_rate: learning rate
        :return: model
        """
        q_net = Sequential()
        q_net.add(tf.keras.Input(shape=state_space))
        q_net.add(Conv2D(32, 8, strides=4, activation="relu"))
        q_net.add(Conv2D(64, 4, strides=2, activation="relu"))
        q_net.add(Conv2D(64, 3, strides=1, activation="relu"))
        q_net.add(Flatten())
        q_net.add(Dense(512, activation="relu"))
        q_net.add(Dense(action_space, activation="linear"))
        q_net.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                      loss='mse')
        q_net.summary()
        return q_net

    def save_model(self):
        """
        Saves model to file system

        :return: None
        """
        tf.saved_model.save(self.q_net, self.model_location)

    def load_model(self):
        """
        Loads previously saved model
        :return: None
        """
        self.q_net = tf.saved_model.load(self.model_location)

    def save_checkpoint(self):
        """
        Saves training checkpoint

        :return: None
        """
        self.checkpoint_manager.save()

    def load_checkpoint(self):
        """
        Loads training checkpoint into the underlying model

        :return: None
        """
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)

    def update_target_network(self):
        """
        Updates the target Q network with the parameters
        from the currently trained Q network.

        :return: None
        """
        if self.verbose != 'none':
            print('Update target Q network')
        self.target_q_net.set_weights(self.q_net.get_weights())

    def train(self, state_batch, next_state_batch, action_batch, reward_batch,
              done_batch, batch_size):
        """
        Train the model on a batch

        :param state_batch: batch of states
        :param next_state_batch: batch of next states
        :param action_batch: batch of actions
        :param reward_batch: batch of rewards
        :param done_batch: batch of done status
        :param batch_size: the size of the batch
        :return: loss history
        """
        state_batch = state_batch.astype(np.float32)
        current_q = self.q_net(state_batch).numpy()
        target_q = np.copy(current_q)
        next_q = self.target_q_net(next_state_batch)
        max_next_q = np.amax(next_q, axis=1)
        for batch_idx in range(batch_size):
            if done_batch[batch_idx]:
                target_q[batch_idx][action_batch[batch_idx]] = \
                    reward_batch[batch_idx]
            else:
                target_q[batch_idx][action_batch[batch_idx]] = \
                    reward_batch[batch_idx] + self.gamma * max_next_q[batch_idx]
        if self.verbose == 'loss':
            print('reward batch shape: ', reward_batch.shape)
            print('next Q shape: ', next_q.shape)
            print('next state batch shape: ', next_state_batch.shape)
            print('max next Q shape: ', max_next_q.shape)
            print('target Q shape: ', target_q.shape)
            print('sample target Q: ', target_q[0])
            print('sample current Q: ', current_q[0])
        history = self.q_net.fit(x=state_batch, y=target_q, verbose=0)
        if self.persist_progress_option == 'all':
            self.save_checkpoint()
        loss = history.history['loss']
        return loss

    def collect_policy(self, state, prev_action, num_eps):
        """
        The policy for collecting data points based on Adjusted Heuristic Exploration.

        :param prev_action: the previous action
        :param state: current state
        :param num_eps: the number of current iteration

        :return: action
        """
        # pylint: disable=no-member
        if (num_eps / self.train_period) % 2 == 0:
            if np.random.random() < self.epsilon:
                return self.follow_heuristic_rule(prev_action, noise=False)
            else:
                return self.follow_heuristic_rule(prev_action, noise=True)
        else:
            return self.opt_policy(state)

    def opt_policy(self, state):
        """
        Outputs a action based on model

        :param state: current state
        :return: action
        """
        state_input = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        action_q = self.q_net(state_input)
        optimal_action = np.argmax(action_q.numpy()[0], axis=0)
        # if self.verbose == 'policy':
        #     print('state: ', state)
        #     print('state_input: ', state_input)
        #     print('action Q: ', action_q)
        #     print('optimal action: ', optimal_action)

        return optimal_action

    def store(self, buffer, state, reward, next_state, action, done):
        buffer.record(state, reward, next_state, action, done)

    def follow_heuristic_rule(self, prev_action, noise):
        """
        Outputs a action based on heuristic neighborhood search

        :param prev_action: the action in the previous state
        :param noise: whether noise will be added

        :return: action
        """

        scale = 2.0

        if prev_action is None:
            prev_action = 0
        if noise:
            action = random.randint(0, self.action_space - 1)
        else:
            action = random.randint(max(0, prev_action - scale), min(prev_action + scale, self.action_space - 1))

        return action
