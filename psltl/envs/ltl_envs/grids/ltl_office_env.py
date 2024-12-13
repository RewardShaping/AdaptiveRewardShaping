from typing import Any
import numpy as np
from psltl.envs.common.grids.office_world import OfficeWorld
from psltl.envs.ltl_envs.grids.ltl_grid_env import LTLGridEnv
from psltl.ltl.partial_sat_atm_load import LoadedPartialSatATM
from psltl.envs.skeletons.env_default_settings import reward_kwargs, setting


class LTLOfficeEnv(LTLGridEnv):
    """
    Attributes
    ----------
    atm: LoadedPartialSatATM
        Automaton parser

    start: tuple
        Starting point        

    map_size: int=1
        Size of the office map        

    max_episode_steps: int=100
        Maximum step per episode        

    reward_kwargs: dict=default_ltl_reward_kwargs
        Reward function information        

    setting: dict=setting
        Environment and measurement setting        

    Methods
    -------
    get_observation(self) -> np.array
        Get observation        

    get_converted_q(self, q: int) -> Any
        Get converted automaton state        
    """
    
    def __init__(
        self,
        atm: LoadedPartialSatATM,
        start: tuple,
        map_size: int = 1,
        max_episode_steps: int = 100,
        action_dim: int = 4, 
        reward_kwargs: dict = reward_kwargs,
        setting: dict = setting
    ):
        env = OfficeWorld(start, map_size)
        # env.reset()
        # env.show()
        super().__init__(env, atm, max_episode_steps, action_dim, reward_kwargs, setting)

    def get_observation(self, q) -> np.array:
        """Get observation concatenated with automaton state

        Returns
        -------
        It will return observation
        """

        converted_q = self.get_converted_q(q)
        if self.setting["node_embedding"]:
            new_obs = []
            for i in range(self.atm.nstates):
                if i == int(q):
                    new_obs += list(self.curr_mdp_state)
                else:
                    new_obs += converted_q

            return np.array(new_obs)
        else:
            return np.append(self.curr_mdp_state, converted_q)