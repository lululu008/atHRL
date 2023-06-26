# Action and Trajectory Planning using Hierarchical Reinforcement Learning （atHRL)

This is the code for the atHRL framework presented in the paper:
Xinyang Lu, Flint Xiaofeng Fan and Tianying Wang. "Action and Trajectory Planning for Urban Autonomous Driving with
Hierarchical Reinforcement Learning."

![Alt text](https://github.com/lululu008/atHRL/blob/main/imgs/workflow.png)

# TLDR

This work introduces a three-level hierarchical reinforcement learning method to handle autonomous driving in complex urban scenarios. It learns to make decisions about the agent’s future trajectory and computes target waypoints under continuous settings based on a hierarchical DDPG algorithm. The waypoints planned by the model are then sent to a low-level controller to generate the steering and throttle commands required for the vehicle maneuver.

## Results

Experiments are conducted in the CARLA simulator with a comparison against several other methods including DDPG, hierarchical DPPG, and hierarchical DQN.

<p align="center">
  <img src=https://github.com/lululu008/atHRL/blob/main/imgs/training.png>
</p>

The results show that atHRL method achieves the highest average reward and average speed among all the methods in the map Town03 in the CARLA simulator.

<p align="center">
  <img src="https://github.com/lululu008/atHRL/blob/main/imgs/reward.png"/>
  <img src="https://github.com/lululu008/atHRL/blob/main/imgs/speed.png"/> 
</p>

In addition, experiments were conducted on the other two maps Town02 and Town04, and show a better performance as well.

<p align="center">
  <img src="https://github.com/lululu008/atHRL/blob/main/imgs/bar_t2.png"/>
  <img src="https://github.com/lululu008/atHRL/blob/main/imgs/bar_t4.png"/> 
</p>


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
