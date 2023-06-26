# Action and Trajectory Planning using Hierarchical Reinforcement Learning （atHRL)

This is the code for the atHRL framework presented in the paper:
Xinyang Lu, Flint Xiaofeng Fan and Tianying Wang. "Action and Trajectory Planning for Urban Autonomous Driving with
Hierarchical Reinforcement Learning."

![Alt text](https://github.com/lululu008/atHRL/blob/main/workflow.png)


## Dependency
### gym-carla

1. Enter the gym-carla folder and install the packages:
```
$ pip install -r requirements.txt
$ pip install -e .
```

2. Download [CARLA_0.9.6](https://github.com/carla-simulator/carla/releases/tag/0.9.6), extract it to some folder, and add CARLA to ```PYTHONPATH``` environment variable:
```
$ export PYTHONPATH=$PYTHONPATH:$YourFolder$/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg
```

### atHRL

1. Enter the code folder and install the packages:
```
$ pip install -r requirements.txt
$ pip install -e .
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
