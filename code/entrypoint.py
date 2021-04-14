"""
Entrypoint (CLI)

This module is responsible for collecting the CLI inputs (the arguments) and
pass them into the underlying modules for execution.
"""
import sys
sys.path.append('/home/xinyang/projects/work/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg')

import argparse
import config
from hdqn_train import train_model
import gym
import gym_carla
import carla
import os
import tensorflow as tf

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

def main():
    """
    The CLI entrypoint to the APIs

    :return: None
    """
    parser = argparse.ArgumentParser(
        description='CLI for Cart Pole DQN with TensorFlow 2.0.')
    parser.add_argument('--mode', dest='mode', type=str,
                        default=config.DEFAULT_MODE,
                        choices=config.MODE_OPTIONS,
                        help='The mode the program should execute.')
    parser.add_argument('--num_iterations', dest='num_iterations', type=int,
                        default=config.DEFAULT_NUM_ITERATIONS,
                        help='The number of episodes the '
                             'agent should train on.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        default=config.DEFAULT_BATCH_SIZE,
                        help='The batch size of training.')
    parser.add_argument('--max_replay_history', dest='max_replay_history', type=int,
                        default=config.DEFAULT_MAX_REPLAY_HISTORY,
                        help='The max length of the replay buffer.')
    parser.add_argument('--gamma', dest='gamma', type=int,
                        default=config.DEFAULT_GAMMA,
                        help='The discount rate.')
    parser.add_argument('--eval_eps', dest='eval_eps', type=int,
                        default=config.DEFAULT_EVAL_EPS,
                        help='Number of episode for evaluation.')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float,
                        default=config.DEFAULT_LEARNING_RATE,
                        help='Learning rate for training.')
    parser.add_argument('--target_network_update_frequency', dest='target_network_update_frequency',
                        type=float, default=config.DEFAULT_TARGET_NETWORK_UPDATE_FREQUENCY,
                        help='How often the target Q network should update.')
    parser.add_argument('--checkpoint_location', dest='checkpoint_location', type=str,
                        default=config.DEFAULT_CHECKPOINT_LOCATION,
                        help='The location for storing training checkpoints.')
    parser.add_argument('--model_location', dest='model_location', type=str,
                        default=config.DEFAULT_MODEL_LOCATION,
                        help='The location for storing pre-trained models.')
    parser.add_argument('--verbose', dest='verbose', type=str,
                        default=config.DEFAULT_VERBOSITY_OPTION,
                        choices=config.VERBOSITY_OPTIONS,
                        help='If logging information should show up in the '
                             'terminal.')
    parser.add_argument('--visualizer_type', dest='visualizer_type', type=str,
                        default=config.DEFAULT_VISUALIZER_TYPE,
                        choices=config.VISUALIZER_TYPES,
                        help='Which type of visualizer to use for training '
                             'progress.')
    parser.add_argument('--render_option', dest='render_option', type=str,
                        default=config.DEFAULT_RENDER_OPTION,
                        choices=config.RENDER_OPTIONS,
                        help='How the game plays should be rendered.')
    parser.add_argument('--persist_progress_option', dest='persist_progress_option',
                        type=str, default=config.DEFAULT_PERSIST_PROGRESS_OPTION,
                        choices=config.PERSIST_PROGRESS_OPTIONS,
                        help='How the training should be persisted.')
    parser.add_argument('--pause_time', dest='pause_time', type=int,
                        default=config.DEFAULT_PAUSE_TIME,
                        help='The time that should pause to start recording.')
    parser.add_argument('--min_steps', dest='min_steps', type=int,
                        default=config.DEFAULT_MIN_STEPS,
                        help='The minimum steps a agent should be evaluate on per episode.')
    args = parser.parse_args()
    max_avg_reward = train_model(
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        max_replay_history=args.max_replay_history,
        gamma=args.gamma,
        eval_eps=args.eval_eps,
        learning_rate=args.learning_rate,
        target_network_update_frequency=args.target_network_update_frequency,
        checkpoint_location=args.checkpoint_location,
        model_location=args.model_location,
        verbose=args.verbose,
        render_option=args.render_option,
        persist_progress_option=args.persist_progress_option,
    )
    print(
        'Final best reward achieved is {0}'
            .format(max_avg_reward))


if __name__ == '__main__':
    main()
