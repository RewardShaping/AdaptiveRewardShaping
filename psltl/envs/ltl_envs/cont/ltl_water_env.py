from typing import Tuple
from gym import spaces
import numpy as np
from psltl.envs.common.cont.water.water_world import WaterWorld
from psltl.envs.ltl_envs.cont.ltl_cont_env import LTLContEnv
from psltl.envs.skeletons.env_default_settings import setting, reward_kwargs
from psltl.ltl.partial_sat_atm_load import LoadedPartialSatATM


class LTLWaterEnv(LTLContEnv):
    """
    Attributes
    ----------
    env: WaterWorld
        WaterWrorld environment continuous space and discrete action

    atm: LoadedPartialSatATM
        Load saved automaton

    max_episode_steps: int
        Maximum number of steps per episode

    reward_kwargs: dict
        Reward function related information containing reward type, adaptive reward shaping

    setting: dict
        Setting for environments such as noise, usage of potential function, or discount factor

    Methods
    -------
    env_intialize(self) -> None:
        Initialize environment setup

    step(self, action: int) -> Tuple:
        Propagate dynamics in the environment

    reset(self) -> np.array:
        Reset environmnet setup to start a new episode
    """

    def __init__(self,
                 env: WaterWorld,
                 atm: LoadedPartialSatATM,
                 max_episode_steps: int = 600,
                 reward_kwargs: dict = reward_kwargs,
                 setting: dict = setting
                 ):
        super().__init__(env, atm, max_episode_steps, reward_kwargs, setting)

        self.setting = setting
        self.rolling_window = setting["rolling"]
        # initialize basic info for env
        self.env_intialize()
        # setup reward function
        self.get_reward_function()

    def env_intialize(self) -> None:
        """Initialize environment setup

        """

        init_state = self.env.reset()
        self.action_space = spaces.Discrete(5)  # noop, up, right, down, left

        if self.setting["use_one_hot"]:
            self.observation_space = spaces.Box(
                low=-2, high=self.atm.nstates, shape=(52 + self.atm.nstates, ), dtype=float)
        elif self.setting["node_embedding"]:
            self.observation_space = spaces.Box(
                low=-2, high=self.atm.nstates, shape=(52 * self.atm.nstates, ), dtype=float)
        else:
            self.observation_space = spaces.Box(
                low=-2, high=self.atm.nstates, shape=(52 + 1, ), dtype=float)

        self.curr_mdp_state = tuple(init_state)
        self.prev_mdp_state = tuple(self.curr_mdp_state)
    