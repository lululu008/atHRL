# Autonomous Driving
## Dependency
### gym-carla

1. Clone the git repo from [cjy1992/gym-carla](https://github.com/cjy1992/gym-carla)
```
$ git clone https://github.com/cjy1992/gym-carla.git
```

2. Enter the repo root folder and install the packages:
```
$ pip install -r requirements.txt
$ pip install -e .
```

3. Download [CARLA_0.9.6](https://github.com/carla-simulator/carla/releases/tag/0.9.6), extract it to some folder, and add CARLA to ```PYTHONPATH``` environment variable:
```
$ export PYTHONPATH=$PYTHONPATH:$YourFolder$/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg
```
