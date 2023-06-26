# Action and Trajectory Planning using Hierarchical Reinforcement Learning （atHRL)

This is the code for the atHRL framework presented in the paper:
Xinyang Lu, Flint Xiaofeng Fan and Tianying Wang. "Action and Trajectory Planning for Urban Autonomous Driving with
Hierarchical Reinforcement Learning."

![Alt text](https://github.com/lululu008/atHRL/blob/main/imgs/workflow.png)

# TLDR

This work introduces a three-level hierarchical reinforcement learning method to handle autonomous driving in complex urban scenarios. It learns to make decisions about the agent’s future trajectory and computes target waypoints under continuous settings based on a hierarchical DDPG algorithm. The waypoints planned by the model are then sent to a low-level controller to generate the steering and throttle commands required for the vehicle maneuver.

## Results

Experiments are conducted in three different maps, Town02, Town03, and Town04 in the CARLA simulator.

![Alt text](https://github.com/lululu008/atHRL/blob/main/imgs/maps.png)

The atHRL method achieves the highest average reward and average speed in Town03.
![alt-text-1](https://github.com/lululu008/atHRL/blob/main/imgs/reward.png) "title-1") ![alt-text-2](https://github.com/lululu008/atHRL/blob/main/imgs/speed.png)


## Installation
1. Setup conda environment
```
$ conda create -n env_name python=3.6
$ conda activate env_name
```

2. Install the gym-carla wrapper (https://github.com/cjy1992/gym-carla) by following the installation steps:

  - Enter the gym-carla folder and install the packages:
  ```
  $ pip install -r requirements.txt
  $ pip install -e .
  ```

  - Download [CARLA_0.9.6](https://github.com/carla-simulator/carla/releases/tag/0.9.6), extract it to some folder, and add CARLA to ```PYTHONPATH``` environment variable:
  ```
  $ export PYTHONPATH=$PYTHONPATH:$YourFolder$/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg
  ```

3. Enter the code folder and install the packages:
```
$ pip install -r requirements.txt
```

## Usage
1. Enter the CARLA simulator folder and launch the CARLA server by:
```
$ ./CarlaUE4.sh -windowed -carla-port=2000
```
You can use ```Alt+F1``` to get back your mouse control.
Or you can run in non-display mode by:
```
$ DISPLAY= ./CarlaUE4.sh -opengl -carla-port=2000
```

2. Enter the code folder of this repo and run:
```
$ ./run_train_eval.sh
```
