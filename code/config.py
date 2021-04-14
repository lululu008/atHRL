"""
Config
"""

DEFAULT_MODE = 'train'
"""
The default mode the program should run in
"""

MODE_OPTIONS = ['train', 'test']
"""
The supported modes
"""

DEFAULT_ENV_NAME = 'carla-v0'
"""
The OpenAI environment name to be used
"""

DEFAULT_NUM_ITERATIONS = 50000
"""
The default number of iteration to train the model
"""

DEFAULT_BATCH_SIZE = 128
"""
The default batch size the model should be trained on
"""

DEFAULT_MAX_REPLAY_HISTORY = 1000000
"""
The default max length of the replay buffer
"""

DEFAULT_GAMMA = 0.95
"""
The default discount rate for the Q learning
"""

DEFAULT_EPSILON = 0.05
"""
The default value for epsilon
"""

DEFAULT_EVAL_EPS = 10
"""
The default number of episode the model should be evaluated with
"""

DEFAULT_LEARNING_RATE = 0.001
"""
The default learning rate
"""

DEFAULT_CHECKPOINT_LOCATION = './checkpoints'
"""
The default location to store the training checkpoints
"""

DEFAULT_MODEL_LOCATION = './model'
"""
The default location to store the best performing models
"""

DEFAULT_TARGET_NETWORK_UPDATE_FREQUENCY = 120
"""
How often the target Q network should get parameter update
from the training Q network.
"""

DEFAULT_RENDER_OPTION = 'none'
"""
The default value for rendering option
"""

RENDER_OPTIONS = ['none', 'collect']
"""
The available render options:

* none: don't render anything
* collect: render the game play while collecting data
"""

DEFAULT_VERBOSITY_OPTION = 'progress'
"""
The default verbosity option
"""

VERBOSITY_OPTIONS = ['progress', 'loss', 'policy', 'init']
"""
The available verbosity options:

* progress: show the training progress
* loss: show the logging information from loss calculation
* policy: show the logging information from policy generation
* init: show the logging information from initialization
"""

DEFAULT_VISUALIZER_TYPE = 'none'
"""
The default visualizer type
"""

VISUALIZER_TYPES = ['none', 'streamlit']

DEFAULT_PERSIST_PROGRESS_OPTION = 'all'
PERSIST_PROGRESS_OPTIONS = ['none', 'all']

DEFAULT_PAUSE_TIME = 0
"""
The default value for pausing before execution starts
to make time for screen recording. It's only available
in testing mode since it's pointless to do so while
training.
"""

DEFAULT_MIN_STEPS = 10
"""
The minimum number of steps the evaluation should run
per episode so that the tester can better visualize how
the agent is doing.
"""

DEFAULT_TRAIN_PERIOD = 2

DEFAULT_DECAY_FACTOR = 0.99

DEFAULT_PENALTY = 1

DEFAULT_CARLA_PARAMS = {
    'number_of_vehicles': 100,
    'number_of_walkers': 0,
    'display_size': 256,  # screen size of bird-eye render
    'max_past_step': 1,  # the number of past steps to draw
    'dt': 0.1,  # time interval between two frames
    'discrete': False,  # whether to use discrete control space
    'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
    'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
    'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
    'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
    'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
    'port': 2000,  # connection port
    'town': 'Town03',  # which town to simulate
    'task_mode': 'random',  # mode of the task, [random, roundabout (only for Town03)]
    'max_time_episode': 1000,  # maximum timesteps per episode
    'max_waypt': 12,  # maximum number of waypoints
    'obs_range': 32,  # observation range (meter)
    'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
    'd_behind': 12,  # distance behind the ego vehicle (meter)
    'out_lane_thres': 2.0,  # threshold for out of lane
    'desired_speed': 8,  # desired speed (m/s)
    'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    'display_route': True,  # whether to render the desired route
    'pixor_size': 64,  # size of the pixor labels
    'pixor': False,  # whether to output PIXOR observation
}

DEFAULT_TRAJECTORY_POINTS = [
    [0.0, 1.0], [0.7071, 0.7071], [1.0, 0.0], [0.7071, -0.7071],
    [0.0, -1.0], [-0.7071, -0.7071], [-1.0, 0.0], [-0.7071, 0.7071],
    [0.0, 2.0], [1.4142, 1.4142], [2.0, 0.0], [1.4142, -1.4142],
    [0.0, -2.0], [-1.4142, -1.4142], [-2.0, 0.0], [-1.4142, 1.4142],
]