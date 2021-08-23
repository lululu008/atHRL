"""This module implements a longitudinal and lateral controller."""

import math
import time
from collections import deque

import numpy as np
import tensorflow as tf


def waypoint_to_control(action_step, current_speed):
    speed_and_point = action_step.action.numpy()[0]
    target_speed = speed_and_point[0]
    waypoint = [speed_and_point[1], speed_and_point[2]]
    pid_acc_controller = PIDLongitudinalController()
    pid_str_controller = PIDLateralController()
    acc = pid_acc_controller.run_step(target_speed, current_speed)
    steer = pid_str_controller.run_step(waypoint)
    control = [acc, steer]
    return tf.constant([control])

class PIDLongitudinalController(object):
    """Implements longitudinal control using a PID.

    Args:
       K_P (:obj:`float`): Proportional term.
       K_D (:obj:`float`): Differential term.
       K_I (:obj:`float`): Integral term.
       dt (:obj:`float`): time differential in seconds.
    """
    def __init__(self,
                 K_P: float = 0.7,
                 K_D: float = 0.005,
                 K_I: float = 0.001,
                 dt: float = 0.1,
                 use_real_time: bool = False):
        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I
        self._dt = dt
        self._use_real_time = use_real_time
        self._last_time = time.time()
        self._error_buffer = deque(maxlen=10)

    def run_step(self, target_speed: float, current_speed: float):
        """Computes the throttle/brake based on the PID equations.

        Args:
            target_speed (:obj:`float`): Target speed in m/s.
            current_speed (:obj:`float`): Current speed in m/s.

        Returns:
            Throttle and brake values.
        """
        error = target_speed - current_speed
        self._error_buffer.append(error)

        if self._use_real_time:
            time_now = time.time()
            dt = time_now - self._last_time
            self._last_time = time_now
        else:
            dt = self._dt
        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / dt
            _ie = sum(self._error_buffer) * dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip(
            (self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -3.0,
            3.0)


class PIDLateralController(object):
    """Implements lateral control using a PID.

    Args:
       K_P (:obj:`float`): Proportional term.
       K_D (:obj:`float`): Differential term.
       K_I (:obj:`float`): Integral term.
       dt (:obj:`float`): time differential in seconds.
    """
    def __init__(self,
                 K_P: float = 0.7,
                 K_D: float = 0.005,
                 K_I: float = 0.001,
                 dt: float = 0.1,
                 use_real_time: bool = False):
        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I
        self._dt = dt
        self._use_real_time = use_real_time
        self._last_time = time.time()
        self._e_buffer = deque(maxlen=10)
        

    def run_step(self, waypoint, yaw=0.0):
        v_begin = [0, 0]

        v_end = [math.cos(math.radians(yaw)), math.sin(math.radians(yaw))]

        v_vec = np.array([v_end[0] - v_begin[0], v_end[1] - v_begin[1], 0.0])
        w_vec = np.array([
            waypoint[0] - v_begin[0], waypoint[1] - v_begin[1],
            0.0
        ])
        _dot = math.acos(
            np.clip(
                np.dot(w_vec, v_vec) /
                (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))

        _cross = np.cross(v_vec, w_vec)

        if _cross[2] < 0:
            _dot *= -1.0

        if self._use_real_time:
            time_now = time.time()
            dt = time_now - self._last_time
            self._last_time = time_now
        else:
            dt = self._dt

        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / dt
            _ie = sum(self._e_buffer) * dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip(
            (self._k_p * _dot) + (self._k_d * _de) + (self._k_i * _ie), -0.3,
            0.3)
