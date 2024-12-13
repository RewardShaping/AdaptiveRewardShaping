import gym
from typing import Any, Tuple
from gym import spaces
import numpy as np
from collections import defaultdict
from psltl.ltl.partial_sat_atm_load import LoadedPartialSatATM
from psltl.envs.skeletons.common_ltl_env import LTLEnv
# call default setup for reward function and environment setup
from psltl.envs.common.grids.craft_world import CraftWorld
from psltl.envs.skeletons.env_default_settings import setting, reward_kwargs


class LTLGridEnv(LTLEnv):
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
        max_episode_steps: int=100, 
        action_dim: int=4, 
        reward_kwargs: dict=reward_kwargs, 
        setting: dict=setting
        ):
        super().__init__(env, atm, max_episode_steps, reward_kwargs=reward_kwargs)

        self.setting = setting
        rolling_window = setting["rolling"]
        self.action_dim = action_dim
        self.measurement_initialize(rolling_window)
        # initialize basic info for env
        self.env_intialize()        
        # setup reward function
        self.get_reward_function()
        
        # to keep track of visitation information for given map
        self.visit_freq = np.zeros(self.shape)
        
        if not self.reward_type in ["progress", "distance", "hybrid", "humman", "success", "naive", "hltlhybrid", "hltlnaive"]:
            raise ValueError("Reward type is not defined properly.")

    def env_intialize(self) -> None:
        """Initialize environment setup
		
		"""

        self.env.reset()
        
        self.shape = (self.env.map_height, self.env.map_width)
        self.action_space = spaces.Discrete(self.action_dim) # up, right, down, left
        if type(self.env.get_features()) == int:
            self.curr_mdp_state = self.env.get_features()
        else:
            self.curr_mdp_state = tuple(self.env.get_features())
        # we will get agent information as a vector of the map
        if self.setting["use_one_hot"]:
            self.observation_space = spaces.Box(low=0, high=max(self.env.map_height, self.env.map_width), 
                                                shape=(np.product(self.shape) + self.atm.nstates, ), dtype=np.uint8)
        elif self.setting["node_embedding"]:
            self.observation_space = spaces.Box(low=0, high=1, shape=(np.product(self.shape) * self.atm.nstates, ), dtype=np.uint8)
            # get vector feauture then (a, b) -> [0, 0, 0, 0, 1, .....] on grid map
            self.curr_mdp_state = tuple(self.env.get_vector_features())
        else:
            # state representation will look like (h, w, q)
            # for tabular Q learning
            if self.env.map_height == 1 or self.env.map_width == 1:
                self.observation_space = spaces.Box(low=0, high=max([self.env.map_height * self.env.map_width, self.atm.nstates]), shape=(2, ), dtype=np.uint8)
            else:
                self.observation_space = spaces.Box(low=0, high=max([self.env.map_height, self.env.map_width, self.atm.nstates]), shape=(3, ), dtype=np.uint8)

        self.prev_mdp_state = self.curr_mdp_state

    # for plot, heat map for exploration check
    def update_visit_frequency(self, mdp_state: Any) -> None:
        """Update visitation frequency for each mdp states on grid maps
		
		Parameters
		----------
		mdp_state: Any
			MDP state, this will be usually vector, or tuple
		"""

        if type(mdp_state) == int:
            mdp_state = (0, mdp_state)
        else:
            mdp_state = tuple(mdp_state)

        self.visit_freq[mdp_state] += 1

    # for plot, heat map for exploration check
    def get_visit_frequency(self) -> np.array:
        """Get visitation frequency and the reset
		
		Returns
		-------
		visit_freq: np.array
            Visitation frequency for grid map, we will use this to plot heatmap
		"""

        visit_freq = self.visit_freq
        self.visit_freq = np.zeros(self.shape)
        
        return visit_freq
    
    def step(self, action: int) -> Tuple:
        """Propagate dynamics in the environment
		
		Parameters
		----------
		action: int
			Action for grid worlds so that it has integer type
        noise: float
            Environmental noise

		Returns
		-------
		It will return (next obervation, reward, done, info)
		"""

        self.total_step += 1
        # will reset once the episode is done
        self.episode_step += 1
        self.prev_mdp_state = self.curr_mdp_state
        self.update_visit_frequency(self.env.get_features())
        
        # 0: up, 1: right, 2: down, 3: left
        if np.random.rand() < float(self.setting["noise"]):
            # up or down 
            if action in [0, 2]:
                action = np.random.choice([1, 3])
            # right or left
            elif action in [1, 3]:
                action = np.random.choice([0, 2])

        # the environment will update all mdp state based on the action taken
        env_state, env_reward, env_done, env_info = self.env.step(action)
        # if we want to use environmental reward, then just return originally designed reward and states without augmentation
        if self.setting["original_env"]:
            return env_state, env_reward, env_done, env_info

        if self.setting["node_embedding"]:
            mdp_state = tuple(self.env.get_vector_features())
        else:
            if type(self.env.get_features()) == int:
                mdp_state = self.env.get_features()
            else:
                mdp_state = tuple(self.env.get_features())

        self.curr_mdp_state = mdp_state
        label = self.env.get_events()
        
        # infeasible case
        if self.setting["missing"]:
            # this is for taxi and office, both case goal label is missing
            if "g" in label:
                label = label.replace("g", "")
        next_q = self.atm.delta(self.curr_q, label)
        next_q = int(list(next_q)[0])

        info = {}

        self.update_q_state(next_q)

        new_obs = self.get_observation(self.curr_q)
        curr_q_rank = self.partial_achieve[self.curr_q]
        reward = self.get_reward(self.prev_q, self.curr_q, self.setting["human"], self.setting["human_designed_reward"])
        
        # highest_success_sofar will be updated on is_terminal part
        done = self.is_terminal()
        
        info.update({
            "curr_q": self.curr_q, "curr_rank": curr_q_rank, "prev_q": self.prev_q, "prev_rank": self.partial_achieve[self.prev_q],
            "label": label, "env_reward": env_reward, "is_success": self.success, "is_partial_success": self.highest_success_sofar,
            "prev_mdp_state": self.prev_mdp_state, "curr_mdp_state": self.curr_mdp_state})
    
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

        self.env.reset()
        self.is_checked = defaultdict(bool)
        self.highest_success_sofar = 0 # rank of the task
        self.success = 0
        self.episode_step = 0
        self.curr_q = 0
        self.prev_q = 0
        if self.setting["node_embedding"]:
            self.curr_mdp_state = tuple(self.env.get_vector_features())
        else:
            if type(self.env.get_features()) == int:
                self.curr_mdp_state = self.env.get_features()
            else:
                self.curr_mdp_state = tuple(self.env.get_features())
        self.prev_mdp_state = self.curr_mdp_state
        init_observation = self.get_observation(self.curr_q)

        return init_observation
    
    # we incorporate automaton state with mdp state
    def get_observation(self, q) -> np.array:
        """Get observation concatenated with automaton state
        
        Returns
        -------
        It will return observation
        """
        
        # convert automaton state here, we may use one-hot encoding here
        converted_q = self.get_converted_q(q)
        observation = np.append(self.curr_mdp_state, converted_q)

        return observation
        
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

        

