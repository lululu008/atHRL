import config
from dqn_agent import DqnAgent
from utils import sigma
import numpy as np

trajectory_points = config.DEFAULT_TRAJECTORY_POINTS


class HierarchicalDqnAgent(object):

    def __init__(self, state_space, action_space, gamma, lr, verbose,
                 checkpoint_location, model_location, persist_progress_option, mode, epsilon, train_period):
        self.agent_behavior = DqnAgent(state_space=state_space,
                                       action_space=action_space,
                                       gamma=gamma, verbose=verbose, lr=lr,
                                       checkpoint_location=checkpoint_location,
                                       model_location=model_location,
                                       persist_progress_option=persist_progress_option,
                                       mode=mode,
                                       epsilon=epsilon,
                                       train_period=train_period,
                                       perception=True
                                       )

        self.agent_trajectory = DqnAgent(state_space=action_space + 1,
                                         action_space=len(trajectory_points),
                                         gamma=gamma, verbose=verbose, lr=lr,
                                         checkpoint_location=checkpoint_location,
                                         model_location=model_location,
                                         persist_progress_option=persist_progress_option,
                                         mode=mode,
                                         epsilon=epsilon,
                                         train_period=train_period,
                                         perception=False
                                         )
        self.train_period = train_period
        self.decay_factor = config.DEFAULT_DECAY_FACTOR

        self._subgoals_behavior = []
        self._subgoals_trajectory = []

        self._curr_behavior = None
        self._curr_trajectory = None
        self._prev_behavior = None
        self._prev_trajectory = None

    def get_behavior_state(self, state):

        return np.copy(state)

    def get_trajectory_state(self, state, curr_behavior):

        trajectory_state = np.array(state)
        trajectory_state = np.append(trajectory_state, curr_behavior)
        return np.copy(trajectory_state)

    def subgoal_completed(self, done, subgoals, subgoal_index):
        # Checks whether the controller has completed the currently specified subgoal.

        return done

    def collect_policy(self, state, num_eps):
        """
        The policy of hierarchical dqn agent

        :param num_eps: the number of current iteration
        :param state: current state

        :return: action
        """

        behavior_state = self.get_behavior_state(state)
        self._curr_behavior = self.agent_behavior.collect_policy(behavior_state, self._prev_behavior, num_eps)

        self._prev_behavior = self._curr_behavior
        trajectory_state = self.get_trajectory_state(state, self._curr_behavior)
        self._curr_trajectory = self.agent_trajectory.collect_policy(trajectory_state, self._prev_trajectory, num_eps)

        self._subgoals_behavior.append(self._curr_behavior)
        self._subgoals_trajectory.append(self._curr_trajectory)

        self._prev_trajectory = self._curr_trajectory
        point = trajectory_points[self._curr_trajectory]

        return self._curr_behavior, point, behavior_state, trajectory_state

    def opt_policy(self, state):
        """
        The policy of hierarchical dqn agent

        :param state: current state
        :return: action
        """

        behavior_state = self.get_behavior_state(state)
        curr_behavior = self.agent_behavior.opt_policy(behavior_state)

        trajectory_state = self.get_trajectory_state(state, curr_behavior)
        curr_trajectory = self.agent_trajectory.opt_policy(trajectory_state)

        point = trajectory_points[curr_trajectory]
        return point

    def store(self, behavior_buffer, trajectory_buffer, behavior_state, trajectory_state,
              reward_behavior, reward_trajectory, next_state, done):

        next_behavior_state = next_state
        self.agent_behavior.store(behavior_buffer, behavior_state, reward_behavior,
                                  next_behavior_state, self._curr_behavior, done)

        next_trajectory_state = np.append(next_state, self._curr_behavior)
        self.agent_trajectory.store(trajectory_buffer, trajectory_state, reward_trajectory,
                                    next_trajectory_state, self._curr_trajectory, done)
        self._curr_behavior = None

    def hybrid_reward(self, done, reward):
        """
        Penalize reward_behavior and reward_trajectory if they fails.

        :param reward: original reward
        :param done: if current state is done
        :return: reward of behavior and reward of trajectory
        """
        penalty = config.DEFAULT_PENALTY

        reward_behavior, reward_trajectory = reward, reward

        for x in self._subgoals_behavior:
            for y in self._subgoals_trajectory:
                if self.subgoal_completed(done, self._subgoals_behavior, x):
                    if self.subgoal_completed(done, self._subgoals_trajectory, y):
                        reward_behavior += penalty
                        reward_trajectory += penalty
                    else:
                        reward_trajectory -= penalty
                else:
                    reward_behavior -= penalty

        return reward_behavior, reward_trajectory

    def update_epsilon(self, reward_eps, eps):
        """
        Outputs the updated epsilon

        :param eps: the number of episode
        :param reward_eps: episode reward
        :return: epsilon
        """
        old_epsilon = self.agent_behavior.epsilon
        if sigma(eps - 4 * self.train_period, eps - 2 * self.train_period, reward_eps) \
                < sigma(eps - 2 * self.train_period, eps, reward_eps):
            new_epsilon = old_epsilon * self.decay_factor
        else:
            new_epsilon = old_epsilon / self.decay_factor

        self.agent_trajectory.epsilon = new_epsilon
        self.agent_behavior.epsilon = new_epsilon

    def save_model(self):
        self.agent_behavior.save_model()
        self.agent_trajectory.save_model()
