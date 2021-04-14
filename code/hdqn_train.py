"""
Training script
"""

import gym
import config
from h_dqn_agent import HierarchicalDqnAgent
from replay_buffer import DqnReplayBuffer
from utils import collect_episode, train_buffer, compute_avg_reward

# pylint: disable=too-many-arguments,too-many-locals
def train_model(
        num_iterations=config.DEFAULT_NUM_ITERATIONS,
        batch_size=config.DEFAULT_BATCH_SIZE,
        max_replay_history=config.DEFAULT_MAX_REPLAY_HISTORY,
        gamma=config.DEFAULT_GAMMA,
        eval_eps=config.DEFAULT_EVAL_EPS,
        learning_rate=config.DEFAULT_LEARNING_RATE,
        target_network_update_frequency=config.DEFAULT_TARGET_NETWORK_UPDATE_FREQUENCY,
        checkpoint_location=config.DEFAULT_CHECKPOINT_LOCATION,
        model_location=config.DEFAULT_MODEL_LOCATION,
        verbose=config.DEFAULT_VERBOSITY_OPTION,
        render_option=config.DEFAULT_RENDER_OPTION,
        persist_progress_option=config.DEFAULT_PERSIST_PROGRESS_OPTION,
        epsilon=config.DEFAULT_EPSILON,
        train_period=config.DEFAULT_TRAIN_PERIOD
):
    """
    Trains a hierarchical DQN agent by playing episodes of the Cart Pole game

    :param epsilon: epsilon is the probability that a random action is chosen
    :param target_network_update_frequency: how frequent target Q network gets updates
    :param num_iterations: the number of episodes the agent will play
    :param batch_size: the training batch size
    :param max_replay_history: the limit of the replay buffer length
    :param gamma: discount rate
    :param eval_eps: the number of episode per evaluation
    :param learning_rate: the learning rate of the back propagation
    :param checkpoint_location: the location to save the training checkpoints
    :param model_location: the location to save the pre-trained models
    :param verbose: the verbosity level which can be progress, loss, policy and init
    :param render_option: if the game play should be rendered
    :param persist_progress_option: if the training progress should be saved
    :param train_period: a predefined number for adjusted heuristic exploration

    :return: (maximum average reward, baseline average reward)
    """
    params = config.DEFAULT_CARLA_PARAMS

    use_epsilon = epsilon
    env_name = config.DEFAULT_ENV_NAME
    train_env = gym.make(env_name, params=params)
    # eval_env = gym.make(env_name, params=params)
    train_env.reset()
    h_agent = HierarchicalDqnAgent(state_space=train_env.observation_space['birdeye'].shape,
                                   action_space=3,
                                   gamma=gamma, verbose=verbose, lr=learning_rate,
                                   checkpoint_location=checkpoint_location,
                                   model_location=model_location,
                                   persist_progress_option=persist_progress_option,
                                   mode='train',
                                   epsilon=use_epsilon,
                                   train_period=train_period
                                   )

    behavior_buffer = DqnReplayBuffer(max_size=max_replay_history)
    trajectory_buffer = DqnReplayBuffer(max_size=max_replay_history)
    max_avg_reward = 0.0
    for eps_cnt in range(num_iterations):
        print("starting a new episode")
        collect_episode(train_env, h_agent, behavior_buffer, trajectory_buffer, render_option, eps_cnt)

        train_buffer(h_agent.agent_behavior, behavior_buffer,
                     batch_size, eps_cnt, target_network_update_frequency)
        train_buffer(h_agent.agent_trajectory, trajectory_buffer,
                     batch_size, eps_cnt, target_network_update_frequency)

        use_eval_eps = eval_eps
        # avg_reward = compute_avg_reward(eval_env, h_agent.opt_policy, num_episodes=use_eval_eps)
        avg_reward = compute_avg_reward(train_env, h_agent.opt_policy, num_episodes=use_eval_eps)
        if avg_reward > max_avg_reward:
            max_avg_reward = avg_reward
            if persist_progress_option == 'all':
                h_agent.save_model()
        if verbose != 'none':
            print(
                'Episode {0}/{1}({2}%) finished with avg reward {3} '
                ' and buffer volume {4}'.format(
                    eps_cnt, num_iterations,
                    round(eps_cnt / num_iterations * 100.0, 2),
                    avg_reward, behavior_buffer.get_volume()))
    train_env.close()
    # eval_env.close()
    return max_avg_reward
