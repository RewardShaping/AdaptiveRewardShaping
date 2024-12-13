import gym
from gym import spaces
import numpy as np
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
from psltl.envs.ltl_envs.cont.ltl_cont_env import LTLContEnv
from psltl.envs.skeletons.env_default_settings import setting, reward_kwargs
from psltl.ltl.partial_sat_atm_load import LoadedPartialSatATM


class MyHalfCheetahEnv(gym.Wrapper):
    def __init__(self):
        # Note that the current position is key for our tasks
        super().__init__(HalfCheetahEnv(exclude_current_positions_from_observation=False))

    def step(self, action):
        # executing the action in the environment
        next_obs, original_reward, env_done, info = self.env.step(action)
        self.info = info
        return next_obs, original_reward, env_done, info

    def get_events(self):
        events = ''
        if self.info['x_position'] < -4:
            events+='b'
        if self.info['x_position'] < -2:
            events+='d'
        if self.info['x_position'] > 2:
            events+='c'
        if self.info['x_position'] > 4:
            events+='e'
        if self.info['x_position'] > 6:
            events+='f'
        if self.info['x_position'] > 8:
            events+='g'
        if self.info['x_position'] > 10:
            events+='a'
        return events


class LTLCheetahEnv(LTLContEnv):
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
                 env: gym.Env,
                 atm: LoadedPartialSatATM,
                 max_episode_steps: int = 1000,
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

        if not self.reward_type in ["progress", "distance", "hybrid", "humman", "origin", "naive", "success"]:
            raise ValueError("Reward type is not defined properly.")

    def env_intialize(self) -> None:
        """Initialize environment setup

        """

        init_state = self.env.reset()
        self.action_space = self.env.action_space

        if self.setting["use_one_hot"]:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18 + self.atm.nstates, ), dtype=float)
        elif self.setting["node_embedding"]:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18 * self.atm.nstates, ), dtype=float)
        else:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18 + 1, ), dtype=float)
            
        self.curr_mdp_state = tuple(init_state)
        self.prev_mdp_state = tuple(self.curr_mdp_state)
    