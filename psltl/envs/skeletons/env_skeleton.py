import gym
from collections import defaultdict, deque


class CommonEnv(gym.Env):
    """
	Attributes
	----------
	max_episode_steps: int
		Every environment must have terminal steps, we cannot run code infinitely many times

	Methods
	-------
	env_intialize(self) -> None:
		Initialize environment condition

	measurement_initialize(self) -> None:
		Intialize variables for measurement

	get_success_rate(self, rolling: int=0) -> float:
		Get success rate

	step(self, action):
        Propagate dynamics

	reset(self):
		Reset environment condition based on terminal conditions

	get_reward_function(self):
		Get reward function such as progress, distance, hybrid

	get_reward(self):
		Get reward from the defined reward function

	is_terminal(self):
		Return True or False considering terminal conditions are met or not
	"""

    def __init__(
        self, 
        max_episode_steps: int
        ):
        self.max_episode_steps = max_episode_steps
    
    def env_intialize(self) -> None:
        raise NotImplementedError

    def measurement_initialize(self, rolling_window: int=20) -> None:
        """Initialize variables to keep track of measurement such as success rate, success, total episode, partial satisfiability
        self.recent_sucess: store recent 100 success over the 100 epsiodes
        self.partial_sat_row: partial sat over row (rank distribution over one episode)
        self.partial_sat_col: partial sat over col (rank distribution over multiple episodes)
        self.is_checked: this is for partial_sat_col. we check only once wether partial sat occured or not for each episode
        """
        self.rolling_window = rolling_window
        self.success = 0
        self.recent_sucess = deque(maxlen=rolling_window)
        self.eval_recent_success = deque(maxlen=rolling_window)
        self.success_rate = []
        self.total_success = 0
        self.total_episode = 0
        self.episode_step = 0
        self.total_step = 0
        self.episode_reward = 0
        self.is_success_observed = False
        # this is to plot
        self.partial_sat_row = defaultdict(int)
        self.partial_sat_col = defaultdict(int)
        self.is_checked = defaultdict(bool)        

    def get_success_rate(self) -> float:
        """Get success rate over epsiodes

		Returns
		-------
		success_rate: float
            Get success rate
		"""

        if self.total_episode == 0:
            return 0

        # window size for measurement
        if self.rolling_window > 0:
            recent_sucess = self.recent_sucess
            if len(recent_sucess) == 0:
                return 0
            
            return sum(recent_sucess) / len(recent_sucess)
        
        # if rolling is zero or negative, we will return caring about all the episodeq
        success_rate = self.total_success / self.total_episode

        return success_rate
        
    def step(self, action):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError
    
    def get_reward_function(self):
        raise NotImplementedError
    
    def get_reward(self):
        raise NotImplementedError
    
    def is_terminal(self):
        raise NotImplementedError