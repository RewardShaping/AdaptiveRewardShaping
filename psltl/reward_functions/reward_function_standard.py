import numpy as np
MAXIMUM_RANK = int(1e4)


class RewardFunction:
	"""
	Attributes
	----------
	atm_kwargs: dict
		This dictionary contains information about automaton such as
		distance function, acceptance states, trapping states, adaptive reward shaping conditions, and so on and so forth.

	Methods
	-------
	update(self):
		Update automaton information related to determination of reward

	update_distance(self, q_trajectory: dict) -> None:
		Update distance function

	get_reward(self):
		Get reward, not implemented
	"""

	def __init__(self, atm_kwargs: dict):
		self.dist = atm_kwargs["dist"]
		self.original_dist = atm_kwargs["origin_dist"].copy()
		self.total_dist = atm_kwargs["total_dist"]
		self.hybrid_eta = atm_kwargs["delta"]
		self.adrs_update = atm_kwargs["adrs_update"]
		self.adrs_mu = atm_kwargs["adrs_mu"]
		self.theta = atm_kwargs["theta"]
		self.rank = atm_kwargs["rank"]
		if self.theta == "dist":
			self.theta = self.total_dist
		self.goal_q = atm_kwargs["goal_q"]
		self.trapping_q = atm_kwargs["trapping_q"]
		self.version = atm_kwargs["version"]
		self.atm = atm_kwargs["atm"]
		
		self.state2next_states = self.atm.state2next_states
		self.update_count = 0
		self.max_update = len(self.rank) - 1
		self.best_progression_so_far = None
		self.updated_progression = []
		self.state2rank = {}
		self.rank2dist = {}
		for q in self.atm.states:
			for idx, (k, v) in enumerate(self.atm.rank.items()):
				if q in v:
					self.state2rank[q] = idx
					self.rank2dist[idx] = self.original_dist[q]
		
	def update(self):
		raise NotImplementedError
		
	def is_max_update(self):
		"""Check number of update exceed pre-defined max update for trajectory
		"""
		return self.max_update <= self.update_count
		
	def update_distance(self, q_trajectory: dict, verbose: int = 0) -> None:
		"""Update distance function based on automaton states trajectory
		
		Parameters
		----------
		q_trajectory: dict
			Information of trajectory of automaton states during the training phase
		verbose: int
			pinrtout distance before udpate, and after update
		"""
		
		margin = np.sum(list(q_trajectory.values()))
		adrs_mu = self.adrs_mu
		theta = self.theta
		self.update_count += 1
		updated = False
		if verbose:
			print("-" * 75)
			print("Update Reward function!")
			print("Distance function before update:", self.dist)
			
		if self.version == 0:	
			for k in self.dist.keys():
				# for the goal automaton state, we do not update anything. The distance will be always 0 on the state
				if k == self.goal_q:
					# self.dist[k] -= theta
					pass
				# for the trapping automaton state, we always give a maximum distance, so that the progress from usual state to the trapping state not to be positive
				elif k == self.trapping_q:
					self.dist[k] = adrs_mu * self.dist[k] + theta
				else:
					self.dist[k] = adrs_mu * self.dist[k] + theta * q_trajectory[k] / margin
			updated = True

		elif self.version == 1:
			# shape of rank example: {0.0: [4], 1.0: [2, 3], 2.0: [0], 15.0: [1]})
			self.visited_q = [k for k, v in q_trajectory.items() if v != 0 and k != self.goal_q and k != self.trapping_q]
			# Find the best progression so far, the smaller, the better
			# This is a rank
			best_progression_so_far = MAXIMUM_RANK
			for rank, (dist, states) in enumerate(self.rank.items()):
				for visited_q in self.visited_q:
					if visited_q in states:
						best_progression_so_far = min(best_progression_so_far, rank)
			
			prev_best_progression = self.best_progression_so_far
			if self.best_progression_so_far == None:
				self.best_progression_so_far = best_progression_so_far
			
			# update best progression if the distance of current progression is close to the acceptance state
			if self.rank2dist[best_progression_so_far] <= self.rank2dist[self.best_progression_so_far]:
				self.best_progression_so_far = best_progression_so_far

			# we update each q states only once because this will be enough as long as the value of theta is big
			if prev_best_progression != self.best_progression_so_far:
				for rank, (dist, states) in enumerate(self.rank.items()):
					if rank >= self.best_progression_so_far:
						for q in states:
							self.dist[q] += theta
				updated = True
			
		if verbose:		
			print("Distance function after update:", self.dist)
			print("Update count:", self.update_count)
			print("-" * 75)
		return updated
		
	def get_reward(self):
		raise NotImplementedError


class Progress(RewardFunction):
	"""
	Attributes
	----------
	atm_kwargs: dict
		This dictionary contains information about automaton such as
		distance function, acceptance states, trapping states, adaptive reward shaping conditions, and so on and so forth.

	Methods
	-------
	update(self, q_trajectory: dict) -> None:
		Update distance function

	get_reward(self, prev_q: int, curr_q: int, human: bool=False, human_designed_reward: dict={}) -> float:
		Get reward based on current automaton information
	"""

	def __init__(self, atm_kwargs: dict):
		super().__init__(atm_kwargs)

	def update(self, q_trajectory: dict, verbose: int = 1) -> None:
		"""Update distance function based on automaton states trajectory
		
		Parameters
		----------
		q_trajectory: dict
			Information of trajectory of automaton states during the training phase
		"""

		self.update_distance(q_trajectory, verbose=verbose)

	def get_reward(self, prev_q: int, curr_q: int, human: bool=False, human_designed_reward: dict={}) -> float:
		"""Description for function get_reward here
		
		Parameters
		----------
		prev_q: int
			Previous automaton state

		curr_q: int
			Current automaton state

		curr_q_rank: int
			Rank of the current automaton state

		human: bool
			If this is True, then we will use human designed reward

		human_designed_reward: dict
			This is dictionary of reward for automaton state based on the rank of the state

		Returns
		-------
		reward: float
			Reward value defined by the automaton states
		"""

		pre_reward = max(float(self.dist[prev_q] - self.dist[curr_q]), 0)
		reward_without_adrs = max(float(self.original_dist[prev_q] - self.original_dist[curr_q]), 0)
		reward = max(pre_reward, reward_without_adrs)

		if human and reward > 0:
			curr_q_rank = self.state2rank[curr_q]
			assert len(human_designed_reward) != 0
			reward = human_designed_reward[curr_q_rank]
		
		return reward


class Hybrid(RewardFunction):
	"""
	Attributes
	----------
	atm_kwargs: dict
		This dictionary contains information about automaton such as
		distance function, acceptance states, trapping states, adaptive reward shaping conditions, and so on and so forth.

	Methods
	-------
	update(self, q_trajectory: dict) -> None:
		Update distance function

	update_hybrid_eta(self) -> None:
		Update delta, which is a trade-off factor to control penalty (from distance function)
		and positive feedback (from progress function)

	get_reward(self, prev_q: int, curr_q: int, human: bool=False, human_designed_reward: dict={}) -> float:
		Get reward based on current automaton information
	"""

	def __init__(self, atm_kwargs: dict):
		super().__init__(atm_kwargs)

	def update(self, q_trajectory: dict, verbose: int = 1) -> None:
		"""Update distance function and delta
		
		Parameters
		----------
		q_trajectory: dict
			Information of trajectory of automaton states during the training phase
		"""

		updated = self.update_distance(q_trajectory, verbose=verbose)
		if updated:
			self.update_hybrid_eta()

	def update_hybrid_eta(self) -> None:
		"""Hybrid reward function has trade-off factor delta to control between positive feedback and penalty
		As we update distance function, the value always increase compared to the original distance, so we need to discoun penalty as the update goes
		"""

		self.hybrid_eta = self.hybrid_eta / self.theta
		
	def get_reward(self, prev_q: int, curr_q: int, human: bool=False, human_designed_reward: dict={}) -> float:
		"""Description for function get_reward here
		
		Parameters
		----------
		prev_q: int
			Previous automaton state

		curr_q: int
			Current automaton state

		curr_q_rank: int
			Rank of the current automaton state

		human: bool
			If this is True, then we will use human designed reward

		human_designed_reward: dict
			This is dictionary of reward for automaton state based on the rank of the state

		Returns
		-------
		reward: float
			Reward value defined by the automaton states
		"""

		if prev_q != curr_q:
			pre_reward = max(float(self.dist[prev_q] - self.dist[curr_q]), 0)
			reward_without_adrs = max(float(self.original_dist[prev_q] - self.original_dist[curr_q]), 0)

			reward = max(pre_reward, reward_without_adrs)
			reward *= (1-self.hybrid_eta)

		else:
			pre_reward = -self.dist[curr_q]
			reward_without_adrs = -self.original_dist[curr_q]

			reward = min(pre_reward, reward_without_adrs) * self.hybrid_eta
		
		if human and reward != 0:
			assert human_designed_reward != None
			curr_q_rank = self.state2rank[curr_q]
			if reward > 0:
				reward = human_designed_reward[curr_q_rank] * (1 - self.hybrid_eta)
			else:
				reward = -human_designed_reward[curr_q_rank] * self.hybrid_eta

		return reward


class NaiveReward(RewardFunction):
	def __init__(self, atm_kwargs: dict):
		super().__init__(atm_kwargs)

	def update(self, q_trajectory: dict, verbose: int = 1) -> None:
		"""Update distance function based on automaton states trajectory

		Parameters
		----------
		q_trajectory: dict
			Information of trajectory of automaton states during the training phase
		"""

		self.update_distance(q_trajectory)

	def get_reward(self, prev_q: int, curr_q: int, human: bool=False, human_designed_reward: dict={}) -> float:
		"""Description for function get_reward here

		Parameters
		----------
		prev_q: int
			Previous automaton state

		curr_q: int
			Current automaton state

		curr_q_rank: int
			Rank of the current automaton state

		human: bool
			If this is True, then we will use human designed reward

		human_designed_reward: dict
			This is dictionary of reward for automaton state based on the rank of the state

		Returns
		-------
		reward: float
			Reward value defined by the automaton states
		"""

		pre_reward = max(float(self.original_dist[prev_q]) - float(self.original_dist[curr_q]), 0)
		reward = 0.
		# we will give a reward 1. for each transition, pre_reward >0 means we made a transition to states close to accepting state
		if pre_reward > 0:
			reward = 1.

		return reward
		

class SuccessReward(RewardFunction):
	def __init__(self, atm_kwargs: dict):
		super().__init__(atm_kwargs)

	def update(self, q_trajectory: dict, verbose: int = 0) -> None:
		"""Update distance function based on automaton states trajectory

		Parameters
		----------
		q_trajectory: dict
			Information of trajectory of automaton states during the training phase
		"""

		self.update_distance(q_trajectory, verbose)

	def get_reward(self, prev_q: int, curr_q: int, human: bool=True, human_designed_reward: float = 0) -> float:
		"""Description for function get_reward here

		Parameters
		----------
		prev_q: int
			Previous automaton state

		curr_q: int
			Current automaton state

		curr_q_rank: int
			Rank of the current automaton state

		human: bool
			If this is True, then we will use human designed reward

		human_designed_reward: dict
			This is dictionary of reward for automaton state based on the rank of the state

		Returns
		-------
		reward: float
			Reward value defined by the automaton states
		"""
		reward = 0.
		# human designed reward can be extrinsic reward such as ctrl reward on Mujoco
		if human:
			reward = human_designed_reward
		# only reward 1000 for reaching goal
		# this is proposed reward in Reward Machine
		if curr_q == self.goal_q:
			reward = 1.

		return reward
	 

class Distance(RewardFunction):
	"""
	Attributes
	----------
	atm_kwargs: dict
		This dictionary contains information about automaton such as
		distance function, acceptance states, trapping states, adaptive reward shaping conditions, and so on and so forth.

	Methods
	-------
	update(self, q_trajectory: dict) -> None:
		Update distance function

	get_reward(self, prev_q: int, curr_q: int, human: bool=False, human_designed_reward: dict={}) -> float:
		Get reward based on current automaton information
	"""

	def __init__(self, atm_kwargs: dict):
		super().__init__(atm_kwargs)

	def update(self, q_trajectory: dict) -> None:
		"""Update distance function based on automaton states trajectory
		
		Parameters
		----------
		q_trajectory: dict
			Information of trajectory of automaton states during the training phase
		"""

		self.update_distance(q_trajectory)

	def get_reward(self, prev_q: int, curr_q: int, human: bool=False, human_designed_reward: dict={}) -> float:
		"""Description for function get_reward here
		
		Parameters
		----------
		prev_q: int
			Previous automaton state

		curr_q: int
			Current automaton state

		curr_q_rank: int
			Rank of the current automaton state

		human: bool
			If this is True, then we will use human designed reward

		human_designed_reward: dict
			This is dictionary of reward for automaton state based on the rank of the state

		Returns
		-------
		reward: float
			Reward value defined by the automaton states
		"""

		pre_reward = -self.dist[curr_q]
		reward_without_adrs = -self.original_dist[curr_q]
		reward = min(pre_reward, reward_without_adrs)

		if human and reward < 0:
			curr_q_rank = self.state2rank[curr_q]
			assert human_designed_reward != None
			reward = human_designed_reward[curr_q_rank]
		
		return reward

