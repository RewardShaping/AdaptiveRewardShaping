import gym
import numpy as np
from psltl.envs.skeletons.env_skeleton import CommonEnv
from collections import defaultdict
from psltl.reward_functions.reward_function_standard import Progress, Hybrid, NaiveReward, SuccessReward
from psltl.ltl.partial_sat_atm_load import LoadedPartialSatATM


class LTLEnv(CommonEnv):
    """
	Attributes
	----------
	env: gym.Env
		Environment the agent will interact

	atm: LoadedPartialSatATM
		Automaton

	max_episode_steps: int
		Maximum episode step for one of terminal conditions

	reward_kwargs: dict
		Reward function related information containing reward type, adaptive reward shaping

	Methods
	-------
	env_intialize(self) -> None:
		Initialize environment setup

	atm_intialize(self) -> None:
		Initialize automaton setup

	get_measurements(self) -> dict:
		Get measurement such as success rate and parital satisfiability

	get_partial_achieve(self, automaton_state: int) -> int:
		Get rank for the given automaton state

	print_current_info(self, distance_check: bool=False) -> None:
		Print current information of automaton, and MDP state extra

	update_partial_sat_col(self, q: int) -> None:
		Keep track of parital satisfiability 

	step(self, action):
		Propagate dynamics in the environment

	get_converted_q(self, q):
		We may convert 'int' type automaton state into 'vector' or 'float'

	reset(self):
		Reset environmnet setup to start a new episode

	reset_q_trajectory(self) -> None:
		Reset trajectory of autoatmon state

	update_q_state(self, next_q: int) -> None:
		Update current automaton state during environment training

	get_reward_function(self) -> None:
		Get one of the reward function, Progress, Hybrid, Distance

	get_reward(self, prev_q: int, curr_q: int, curr_q_rank: int, human: bool=False, human_designed_reward: dict={}) -> float:
		Get reward based on automaton state

	get_atm_kwargs(self) -> dict:
		Get automaton information

	is_terminal(self) -> bool:
		Determine the termination of the current episode
	"""

    def __init__(
        self, 
        env: gym.Env, 
        atm: LoadedPartialSatATM, 
        max_episode_steps: int, 
        reward_kwargs: dict
        ):
        super().__init__(max_episode_steps)
        """
        env: mujoco env, or custom env
        atm: automaton parser
        max_episode_steps: maximum steps per episode, if exceed the number, env will be terminated
        reward_kwargs: key values = {reward_type, adaptive_rs, adrs_delta, adrs_update}
        """
        
        self.env = env
        self.atm = atm

        self.reward_type = reward_kwargs["reward_type"]
        self.adaptive_rs = reward_kwargs["adaptive_rs"]
        self.adrs_update = reward_kwargs["adrs_update"]
        self.hybrid_eta = reward_kwargs["hybrid_eta"]
        self.adrs_mu = reward_kwargs["adrs_mu"]
        self.theta = reward_kwargs["theta"]
        self.version = reward_kwargs["version"]
        self.rank = self.atm.rank

        # env initialize should initialize env.reset(), mdp state...
        # atm initialize should initialize atm conditions ...
        
        self.atm_intialize()

    def env_intialize(self) -> None:
        raise NotImplementedError

    def atm_intialize(self) -> None:
        """Initialize automaton information
        Distance function, Original Distance function (Note that distance function will be updated if we use adpative reward shaping)
        Rank for each automaton state, total rank, means number of different rank
        Acceptance state, or trapping state
		
		"""

        # distance function for automaton
        self.dist = self.atm.distances
        # original distance funnction for automaton
        self.original_dist = self.atm.distances.copy()
        # based on distance, we make a class, we have two different automaton state 0, 1 but if they have the same distnace 5, 5, {0, 1} = same class
        self.rank = self.atm.rank
        # the number of different ranks
        self.total_rank = self.atm.total_rank
        # current automaton state based on environment interation
        self.curr_q = 0 # initial automaton state
        # previous one 
        self.prev_q = self.curr_q
        # goal automaton states
        self.goal_q = int(list(self.atm.acc)[0])
        # trapping state will generate one more terminal condition
        self.no_trapping_q = False

        # try case, if there is no trapping q, then this will trigger exception
        if len(list(self.atm.trapping_q)) > 0:
            self.trapping_q = int(list(self.atm.trapping_q)[0])
        else:
            # case there is no trapping state
            self.no_trapping_q = True
            self.trapping_q = -np.inf

        # distinguish non acc states with trapping state and goal state
        if type(self.trapping_q) == int:
            self.non_acc_states = self.atm.states - {self.trapping_q, self.goal_q}
        else:
            self.non_acc_states = self.atm.states - {self.goal_q}

        # for reward shaping, we use total distance of the automaton
        # total distance means the sum of all distances on automaton states
        self.total_dist = np.sum(list(self.dist.values()))
        # this is for adrs, to update distance of automaton states
        # keep track of automaton states frequency during training, based on this values, we define pdf and use the probability density function to update distance
        self.q_trajectory = defaultdict(int)
        self.q_trajectory[self.curr_q] += 1
        self.q_trajectory[self.goal_q]
        
        # set up partial achievement map
        self.partial_achieve = {}
        for automaton_state in self.atm.states:
            self.partial_achieve[automaton_state] = self.get_partial_achieve(automaton_state)

    def get_measurements(self) -> dict:
        """Get measurement success rate, partial satisfibability, and extra
		
		Returns
		-------
		info: dict
            Information about reward type, usage of adaptive reward shaping, episode step, so on and so forth
		"""

        info = {}
        info["Reward type"] = self.reward_type
        info["Adapative reward shaping"] = self.adaptive_rs
        info["Episode step"] = self.episode_step
        # partial satisfiability histogram
        # this will store integer value, how many times the automaton states have visited
        info["Partial Sat"] = self.partial_sat_row
        # we will normalize below
        # for partial sat column for percentage, we will divide the value by the total episode
        # TODO: get rolling window as variable 
        normalized_partial_sat_col = {}
        for key, value in self.partial_sat_col.items():
            if self.total_episode == 0:
                normalized_partial_sat_col[key] = 0
            else:
                normalized_partial_sat_col[key] = float(value) / float(self.total_episode)

        info["Partial Sat Column"] = normalized_partial_sat_col
        if len(self.success_rate) > 0:
            info["Success rates"] = self.success_rate

        return info
    
    def get_partial_achieve(self, automaton_state: int) -> int:
        """Get partial achievement based on rank for automaton states
        We have defined some rank on automaton states
        Based on the rank for each automaton state, we will measure partial achievement of each envrionment
		
		Parameters
		----------
		automaton_state: int
			Automaton state
		Returns
		-------
		rank: int
            Rank of the automaton state based on distance
		"""
        for idx, key_value in enumerate(self.rank.items()):
            _, value = key_value
            if automaton_state in value:
                rank = self.total_rank - idx
                return rank
            
    # printout some info
    def print_current_info(self, distance_check: bool=False) -> None:
        """Print out current information of automaton
		
		Parameters
		----------
		distance_check: bool
			Decide printing out distance function or not
		"""

        print("=" * 75)
        print('Current automaton state: ', self.curr_q)
        print('Previous automaton state: ', self.prev_q)
        print('Goal automaton state: ', self.goal_q)
        if not self.no_trapping_q:
            print('Trapping automaton state: ', self.trapping_q)
        else:
            print('There is no trapping state')
        print('Current MDP States: ', self.curr_mdp_state)
        print('Previous MDP States: ', self.prev_mdp_state)
        if distance_check:
            print("Current distance function is: ", self.dist)
        print("=" * 75)

    # to plot    
    def update_partial_sat_col(self, q: int) -> None:
        """Keep track of partial satisfiability 
		
		Parameters
		----------
		q: Automaton state we have observed
			
		"""

        partial_achieve = self.partial_achieve[q]
        self.partial_sat_col[partial_achieve] += 1

    def step(self, action):
        raise NotImplementedError

    def get_converted_q(self, q):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError
    
    def reset_q_trajectory(self) -> None:
        """Once we update distance function with automaton state trajectory, we will reset automaton state trajectory and recollect
		
		"""
        
        self.q_trajectory = defaultdict(int)

    # update q state interacting with environment
    def update_q_state(self, next_q: int) -> None:
        """Update automaton state trajectory and current, previous automaton state based on environment interation
		
		Parameters
		----------
		next_q: int
			We only need next_q, which means update old automaton state by next automaton state
		"""

        self.prev_q = self.curr_q
        self.curr_q = next_q
        self.q_trajectory[next_q] += 1

    def get_reward_function(self) -> None:
        """Get reward function, there are three cases, Progress, Distance, Hybrid

		"""

        atm_kwargs = self.get_atm_kwargs()
        
        if self.reward_type == "progress":
            reward_function = Progress(atm_kwargs)
        elif self.reward_type == "naive":
            reward_function = NaiveReward(atm_kwargs)
        elif self.reward_type == "hybrid":
            reward_function = Hybrid(atm_kwargs)
        elif self.reward_type == "hltlnaive":
            reward_function = HLTLRewardFunction(atm_kwargs)
        elif self.reward_type == "hltlhybrid":
            reward_function = HLTLHybrid(atm_kwargs)
        elif self.reward_type == "success":
            reward_function = SuccessReward(atm_kwargs)
        else:
            raise ValueError("Please check the reward function type, we only provide progress, distance, hybrid, eval, origin, success reward functions")
        self.reward_function = reward_function

    def get_reward(self, prev_q: int, curr_q: int, human: bool=False, human_designed_reward: dict={}) -> float:
        """Description for function get_reward here
		
		Parameters
		----------
		prev_q: int
			Previous automaton state

		curr_q: int
			Current automaton state

		curr_q_rank: int
			Rank of the current automaton state by distance

		human: bool
			Do we use human to design rewrad?

		human_designed_reward: dict
			Human deisned reward, but reawrd signal will be given with the same logic of LTL reward shaping just exact value might be different 

		Returns
		-------
		reward: float
            Reward based on defined reward function
		"""

        reward = self.reward_function.get_reward(prev_q, curr_q, human, human_designed_reward)
        return reward

    # getting information for automaton state
    def get_atm_kwargs(self) -> dict:
        """Get automaton information
		
		Returns
		-------
		atm_kwargs:dict
            Dictionary which contains information of automaton such as reward type, current automaton states, or related to adaptive reward shaping
            This information will be used to get reward and determine terminal condition
		"""

        atm_kwargs = dict([
            ("reward_type", self.reward_type), ("adrs_mu", self.adrs_mu),
            ("goal_q", self.goal_q), ("trapping_q", self.trapping_q),
            ("total_dist", self.total_dist), ("theta", self.theta),
            ("adrs_update", self.adrs_update), ("origin_dist", self.original_dist),
            ("dist", self.dist), ("delta", self.hybrid_eta),
            ("version", self.version), ("rank", self.rank), ("atm", self.atm)
        ])

        return atm_kwargs
    
    def get_counter_factual_exp(self, q: int, label: str):
        counter_factual_experiences = []
        imagined_qs = self.atm.non_terminal_states
        
        # for automaton state q, it has been already stored in the transition
        for imagined_q in imagined_qs - {q}:
            # counter factual
            cf_obs = self.get_observation(imagined_q)
            imagined_next_q = list(self.atm.delta(imagined_q, label))[0]
            imagined_rank = self.partial_achieve[imagined_next_q]
            imagined_reward = self.get_reward(imagined_q, imagined_next_q, imagined_rank, self.setting["human"], self.setting["human_designed_reward"])
            imagined_done = False
            if imagined_next_q in self.atm.acc or imagined_next_q in self.atm.trapping_q:
                imagined_done = True
            counter_factual_experiences.append((cf_obs, imagined_reward, imagined_done))

        return counter_factual_experiences
    
    def is_terminal(self) -> bool:
        """Check terminal conditions. For LTL environments, there are three conditions.
        1. Current automaton state belongs to the set of acceptance states
		2. Current automaton state belongs to the set of trapping states (Optional: trapping state may not exists e.x.) Eventually reach goal )
        3. Agents reached maximum episode step of the environment

		Returns
		-------
		done: bool
            True or False based on the above three or two conditions
		"""

        done = False
        self.highest_success_sofar = self.partial_achieve[self.curr_q]

        # trapping state does not trigger the end of epsiode
        if self.setting["violation_end"]:
            reach_trapping_state = (self.curr_q == self.trapping_q)
        else:
            reach_trapping_state = False
        
        reach_goal_state = (self.curr_q == self.goal_q)
        if reach_goal_state:
            self.is_success_observed = True

        exceed_max_episode_steps = (self.episode_step >= self.max_episode_steps)
        
        # there is a trapping state
        if not self.no_trapping_q:
            predicate = reach_trapping_state or reach_goal_state or exceed_max_episode_steps
        # if no trapping state
        else:
            predicate = reach_goal_state or exceed_max_episode_steps
        
        if predicate:
            done = True
            self.total_episode += 1
            use_adaptive_reward_shaping = (self.adaptive_rs)
            adrs_update_interval = (self.total_episode % self.adrs_update == 0)

            if self.curr_q == self.goal_q:
                self.success = 1
                self.total_success += 1
            else:
                self.success = 0

            self.recent_sucess.append(self.success)
            # 0.1 is hyperparameter
            is_enough_success = (sum(self.recent_sucess) / len(self.recent_sucess)) >= 0.1
            update_distance_function = use_adaptive_reward_shaping and adrs_update_interval and (not is_enough_success) and not self.reward_function.is_max_update()
            
            if update_distance_function:
                self.reward_function.update(self.q_trajectory, verbose=1)
                # self.reset_q_trajectory()
                # exit("line 418 of common ltl env")
            
            # if hasattr(self.reward_function, "hltl_update") and adrs_update_interval:
            # if hasattr(self.reward_function, "hltl_update"):
                # self.reward_function.update(self.q_trajectory, verbose=0)
                # self.reward_function.hltl_update()

            # we count partial success except for violation
            # for example, if the agent ends up with trappinp state achieving rank 3 partial success before the trapping state,
            # we count rank 3 task as the last partial success
            if reach_trapping_state:
                self.highest_success_sofar = self.partial_achieve[self.prev_q]
            
        return done

    def is_success(self, curr_q: int):
        is_success = False
        if curr_q == self.goal_q:
            is_success = True
        return is_success