from GolfField import GolfField

from abc import ABC, abstractmethod
import gym
from gym import spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numbers
import numpy as np
from typing import Callable, List, Tuple, Union


class GolfEnv():
    max_step: int = 100
    power_limit: float = 30

    def __init__(self, seed: int, obstacles_num: int):
        self.field = GolfField(seed, obstacles_num)
        self.observation_space = spaces.Box(low=0, high=self.field.field_size, shape=(2,), dtype=np.float32)

    def reset(self) -> Tuple[np.ndarray, dict]:
        gameball, info = self.field.reset()
        return gameball.center.p, info

    def step(self, action: Tuple[float, float]):
        # Takes action with angle in [0, 360) and power in (0, 30]
        angle, power = action
        assert power < self.power_limit, 'power limit is {self.power_limit}'


        old_dist_to_win = (self.field.hole.center - self.field.gameball.center).len()

        observation, reward, done, truncated, info = self.field.step((angle, power))

        new_dist_to_win = (self.field.hole.center - self.field.gameball.center).len()
        done = done or self.field.current_step >= self.max_step  # Checks if maximum steps reached.

        return observation.center.p, reward - new_dist_to_win, done, truncated, info

    def render(self, ax = None, mode = 'human'):
        self.field.render(user_ax=ax)

    def render_wind(self, func_for_action: Callable[[float, float], Tuple[float, float]], user_ax = None, image_path = 'temp_wind.png'):
        self.field.render_wind(func_for_action, user_ax, image_path)

    def close(self):
        pass


class GolfEnv8d1p(GolfEnv):
    def __init__(self, seed: int, obstacles_num: int):
        super().__init__(seed, obstacles_num)
        self.action_space = spaces.Discrete(8)

    def step(self, action: int):
        angle_8 = action
        assert 0 <= angle_8 < 8 and isinstance(angle_8, numbers.Number), \
               f'{self.__class__.__name__}.step() takes an integer in the range [0, 7]'
        angle_360 = (angle_8 / 8) * 360

        return super().step((angle_360, 4))
