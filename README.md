# Autonomous Driving
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

### interp_e2e_driving

1. Enter the code folder and install the packages:
```
$ pip install -r requirements.txt
$ pip install -e .
