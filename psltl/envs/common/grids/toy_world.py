from typing import Tuple, Any
import numpy as np 
import gym
from psltl.envs.common.grids.game_objects import *


class ToyWorld(gym.Env):
    """
    Attributes
    ----------
    left: int=10
        Most left location of the map

    right: int=10
        Most right location of the map

    Methods
    -------
    execute_action(self, a: int) -> None
        Execute action in the game
        
    _get_new_position(self, x: int, a: int) -> int
        Get new position
        
    step(self, action: int) -> Tuple
        Propagate dynamcis

    reset(self) -> np.array
        Reset the game

    get_label(self) -> dict
        Get label map
        
    get_features(self) -> int
        Get features
        
    get_vector_features(self) -> np.array
        Get vectorized features
    """

    def __init__(
        self, 
        left: int=10,
        right: int=10,
        ):

        # define a 1-D observation space
        self.left = left
        self.right = right
        self.shape = self.left + self.right + 1

        # define height and width to make it compatiable with office, craft...
        self.map_height = 1 # 1-D observation space
        self.map_width = self.shape
        
        # label map
        self.label = self.get_label()

        # robot position
        self.pos = 0

    def execute_action(self, a: int) -> None:
        """We execute 'action' in the game
        
        Parameters
        ----------
        a: int
            Action taken
            
        """

        x = self.pos
        self.pos = self._get_new_position(x, a)

    def _get_new_position(self, x: int, a: int) -> int:
        """Description for function _get_new_position here
        
        Parameters
        ----------
        x: int
            Current position
        
        a: int
            Action taken
            
        Returns
        -------
        x: int
            Next position taken by the action
        """

        action = TwoActions(a)
        # executing action
        if action == TwoActions.left : x -= 1
        if action == TwoActions.right: x += 1

        return x

    def step(self, action: int) -> Tuple:
        """Propagate dynamcis
        
        Parameters
        ----------
        action: int
            Action taken
            
        Returns
        -------
        next obs, reward, done, info: Tuple
            Experience we would get
        """

        self.episode_step += 1
        # left action
        if action == 0:
            self.pos -= 1
        # right action
        else:
            self.pos += 1
        
        # reward is not defined
        # The reward will be designed by LTL 
        reward = 0 
        done = False
        info = {}
        
        self.pos = min(self.pos, self.right)
        self.pos = max(self.pos, -self.left)
        
        return np.array([self.pos]), reward, done, info
        
    def reset(self) -> np.array:
        """Reset environment
        
        Returns
        -------
        init_pos: np.array
            Initial position
        """

        self.pos = 0
        self.episode_step = 0
        init_pos = np.array([self.pos])

        return init_pos
    
    def get_label(self) -> dict:
        """Get label map
        
        Returns
        -------
        label: dict
            Label map
        """

        label = {}
        for pos in range(-11, 11):
            if pos == -3:
                label[pos] = "b"
            elif pos == 6:
                label[pos] = "a"
            elif pos == 10:
                label[pos] = "b"
            else:
                label[pos] = ""

        return label
    
    def get_events(self) -> str:
        """Get events
        
        Returns
        -------
        label: str
            current label
        """

        label = self.label[self.pos]

        return label
    
    def get_features(self) -> int:
        """Get env features
        
        Returns
        -------
        Current robot position
        """

        return self.pos
    
    def get_vector_features(self) -> np.array:
        """Get vector features
        
        Returns
        -------
        ret: np.array
            Vectorized features will have 1 only for the robot position
        """

        ret = np.zeros(self.shape, dtype=np.float64)
        ret[self.pos] = 1

        return ret