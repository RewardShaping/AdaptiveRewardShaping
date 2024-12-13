from typing import Callable
import numpy as np


def adjusting_schedule(initial_value: float, min_value: float=1e-4, version: int=0) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        if version == 0:
          return min_value + np.exp(-(1-progress_remaining) * 25) * (initial_value - min_value)
        elif version == 1:
          return initial_value +(1-progress_remaining) * (min_value - initial_value)
        
    return func