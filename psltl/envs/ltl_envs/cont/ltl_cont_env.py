import gym
from typing import Any, Tuple
import numpy as np
from psltl.envs.common.cont.water.water_world import WaterWorld
from psltl.ltl.partial_sat_atm_load import LoadedPartialSatATM
from psltl.envs.skeletons.common_ltl_env import LTLEnv


class LTLContEnv(LTLEnv):
    """
	Attributes
	----------
	env: gym.Env
        Environment the agent will interact with

	atm: LoadedPartialSatATM
		Automaton 

	max_episode_steps: int=100
		Maximum episode step for one of terminal conditions

	reward_kwargs: dict=default_reward_kwargs
        Reward function related information containing reward type, adaptive reward shaping

	setting: dict=setting
		Setting for environment such as use one hot encoding or use vectorized feature or for some measurement

	Methods
	-------
	env_intialize(self) -> None:
		Initialize environment setup

	update_visit_frequency(self, mdp_state: Any) -> None:
		Update visitation frequency of each MDP state on grid maps

	get_visit_frequency(self) -> np.array:
		Get visitation frequency we have recorded and reset

	get_observation(self) -> np.array:
		Get new observation based on current MDP state and automaton state

	step(self, action: int) -> Tuple:
		Propagate dynamics in the environment

	reset(self) -> np.array:
		Reset environmnet setup to start a new episode

	get_converted_q(self, q: int) -> Any:
		We may convert 'int' type automaton state into 'vector' or 'float'
	"""

    def __init__(
        self, 
        env: gym.Env, 
        atm: LoadedPartialSatATM,
        max_episode_steps: int, 
        reward_kwargs: dict, 
        setting: dict
        ):
        super().__init__(env, atm, max_episode_steps, reward_kwargs)

        self.setting = setting
        rolling_window = setting["rolling"]
        self.measurement_initialize(rolling_window)
        # setup reward function
        self.get_reward_function()
        
        if not self.reward_type in ["progress", "distance", "hybrid", "human", "origin", "naive", "success"]:
            raise ValueError("Reward type is not defined properly.")

    def env_intialize(self) -> None:
        raise NotImplementedError

    # we incorporate automaton state with mdp state
    def get_observation(self, q) -> np.array:
        converted_q = self.get_converted_q(q)
        if not self.setting["node_embedding"]:
            # convert automaton state here, we may use one-hot encoding here
            observation = np.append(self.curr_mdp_state, converted_q)
        else:
            observation = []
            for i in range(self.atm.nstates):
                if i == int(q):
                    observation += list(self.curr_mdp_state)
                else:
                    observation += converted_q

        return observation
    
    def step(self, action: int) -> Tuple:
        """Propagate dynamics in the environment

                Parameters
                ----------
                action: int
                        Action for grid worlds so that it has integer type

                Returns
                -------
                It will return (next obervation, reward, done, info)
                """

        self.total_step += 1
        self.episode_step += 1

        self.prev_mdp_state = self.curr_mdp_state
        # 0: up, 1: right, 2: down, 3: left
        if np.random.rand() < float(self.setting["noise"]):
            # for water world
            if isinstance(self.env, WaterWorld):
                # up or down 
                if action in [0, 2]:
                    action = np.random.choice([1, 3])
                # right or left
                elif action in [1, 3]:
                    action = np.random.choice([0, 2])
            # for half cheetah environment
            else:
                action += np.random.uniform(-0.1, 0.1)
                action = np.clip(action, -1., 1.)

        mdp_state, mdp_reward, done, info = self.env.step(action)
        reward_ctrl = info.get('reward_ctrl')
        self.curr_mdp_state = tuple(mdp_state)

        label = self.env.get_events()
        self.label = label
        # something is missing, so task achievement is infeasible
        if self.setting["missing"]:
            # for water
            if "c" in label and self.curr_q == 3:
                label = label.replace("c", "")
            # for cheetah
            if "a" in label and self.curr_q == 4:
                label = label.replace("a", "")
        
        next_q = self.atm.delta(self.curr_q, label)
        next_q = int(list(next_q)[0])
        info = {}
            
        self.update_q_state(next_q)

        new_obs = self.get_observation(self.curr_q)
        curr_q_rank = self.get_partial_achieve(self.curr_q)
        reward = self.get_reward(self.prev_q, self.curr_q, self.setting["human"], mdp_reward)
        
        done = self.is_terminal()
        
        info.update({"curr_q": self.curr_q, "curr_rank": curr_q_rank, "prev_q": self.prev_q, "prev_rank": self.partial_achieve[self.prev_q],
                "label": label, "env_reward": mdp_reward, "is_success": self.success, "is_partial_success": self.highest_success_sofar})

        return new_obs, reward, done, info

    def reset(self) -> np.array:
        """Reset environment setup

                Returns
                -------
                init_observation: np.array
            Initial observation after resetting the environment
                """

        if self.curr_q in self.atm.trapping_q:
            self.update_partial_sat_col(self.prev_q)
        else:
            self.update_partial_sat_col(self.curr_q)

        self.episode_step = 0
        self.curr_q = 0
        self.prev_q = 0
        self.success = 0
        self.label = ""

        init_mdp_state = self.env.reset()
        self.curr_mdp_state = tuple(init_mdp_state)
        self.prev_mdp_state = self.curr_mdp_state
        init_observation = self.get_observation(self.curr_q)

        return init_observation

    
    def get_converted_q(self, q: int) -> Any: 
        """We may convert 'int' type automaton state into 'vector' or 'float'
		
		Parameters
		----------
		q: int
            Automaton state
        
		Returns
		-------
		converted_q: Any
            This could be vector value, or float, or whatever you want to convert the integer value into other
            In this case, vector (one hot encoding) or (original value)
		"""
        
        one_hot = np.identity(self.atm.nstates)
        if self.setting["use_one_hot"]:
            converted_q = one_hot[q]
        elif self.setting["node_embedding"]:
            converted_q = [0. for _ in range(len(self.curr_mdp_state))]
        else:
            converted_q = int(q)

        return converted_q

        

