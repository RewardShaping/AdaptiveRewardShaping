from typing import List, Dict
import torch as th
from stable_baselines3.common.utils import get_linear_fn


class GridWorldLearningParams:
    def __init__(
        self,
        # algorithm related
        algo_name: str = "dqn", 
        policy_kwargs: Dict[str, List[int]] = {"net_arch": [], "with_bias": False, "optimizer_class": th.optim.SGD}, # first 4 is embedding layer
        learning_starts: int = 1,
        train_freq: int = 1,
        gradient_steps: int = 1,
        target_update_interval: int = 1,
        buffer_size: int = 50000,
        batch_size: int = 32,
        max_grad_norm: float = 1, 
        # exploration related
        exploration_fraction: float = 0.8,
        exploration_initial_eps: float = 0.2,
        exploration_final_eps: float = 0.1,
        # learning related
        learning_rate_start: float = 1e-1,
        learning_rate_end: float = 1e-5,
        learning_fraction: float = 0.4, 
        # reward function related
        use_adrs: List[bool] = [False,],
        reward_types: list = ["progress", "hybrid", "distance"],
        adrs_mu: float = 0.5,
        adrs_update: int = 10000,
        theta = "dist",
        hybrid_eta: float = 0.001,
        version: int = 1, 
        # env related
        episode_step: int = int(1e2),
        env_name: str = "office",
        gamma: float = 0.9,
        vector: bool = False,
        node_embedding: bool = False,
        use_one_hot: bool = False,
        use_noise: bool = False,
        noise_level: float = 0,
        map_size: int = 1,
        map_id: int = 0,
        violation_end: bool = False, 
        missing: bool = False,
        human: bool = False,
        # training related
        total_timesteps: int = int(1e4),
        total_run: int = 6,
        seed: int = 0,
        # evaluation related
        eval_freq: int = 100,
        rolling: int = 20,
        init_qs: dict = {"distance": 2, "hybrid": 2, "progress": 2},
        save: bool = True,
        # counter factual exp
        cf: bool = False,
    ):
        # reward shaping related
        self.reward_types = reward_types
        self.use_adrs = use_adrs
        self.adrs_update = adrs_update
        self.adrs_mu = adrs_mu
        self.hybrid_eta = hybrid_eta

        # evaluation related
        self.eval_freq = eval_freq
        self.rolling = rolling # rolling window size for evaluation

        # environment related
        self.episode_step = episode_step
        self.use_one_hot = use_one_hot
        self.env_name = env_name
        self.gamma = gamma
        self.vector = vector # use vector representation for state space
        self.node_embedding = node_embedding
        self.use_noise = use_noise
        self.noise_level = noise_level
        self.map_size = map_size
        self.map_id = map_id
        self.seed = seed
        self.violation_end = violation_end
        self.missing = missing
        self.human = human
        self.theta = theta
        self.version = version
        
        # exploration related
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_fraction = exploration_fraction
        self.exploration_final_eps = exploration_final_eps

        # learning related
        self.gamma = gamma
        self.learning_rate_start = learning_rate_start
        self.learning_rate_end = learning_rate_end
        self.learning_fraction = learning_fraction

        # total run and timesteps
        self.total_run = total_run
        self.total_timesteps = total_timesteps

        # algorithm setup related
        self.algo_name = algo_name
        self.policy_kwargs = policy_kwargs
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        # algorithm learning related
        self.max_grad_norm = max_grad_norm
        self.train_freq = train_freq
        self.learning_starts = learning_starts
        self.gradient_steps = gradient_steps
        self.target_update_interval = target_update_interval
        self.init_qs = init_qs
        self.cf = cf
        self.save = save

        self.set_learning_rate()

    def set_learning_rate(self):
        self.learning_rate = get_linear_fn(self.learning_rate_start, self.learning_rate_end, self.learning_fraction)

    def print_infos(self):
        print("=" * 75)

        print("Env: ", self.env_name)
        print("Learning rate: ", self.learning_rate)
        print("Exploration final eps: ", self.exploration_final_eps)
        print("Gamma: ", self.gamma)
        print("Total Timesteps: ", self.total_timesteps)
        print("Total Runs: ", self.total_run)
        print("Reward Types: ", self.reward_types)
        print("Usage of Adaptive Reward Shaping: ", self.use_adrs)
        if True in self.use_adrs:
            print("Adaptive Reward Shaping Gamma: ", self.hybrid_eta)
        print("Epsiode End Step: ", self.episode_step)
        print("Buffer Size: ", self.buffer_size)
        print("Train Frequency: ", self.train_freq)
        print("Batch Size: ", self.batch_size)

        print("=" * 75)

    def model_params(self, algo_name):
        assert algo_name in ["dqn", "ddqn"]
        # for grid world example, we use linear dqn structure as default
        if algo_name == "dqn" or algo_name == "ddqn":
            params = {
                "policy_kwargs": self.policy_kwargs,
                "tensorboard_log": None,
                "gamma": self.gamma,
                "learning_rate": self.learning_rate,
                "train_freq": self.train_freq,
                "batch_size": self.batch_size,
                "buffer_size": self.buffer_size,
                "learning_starts": self.learning_starts,
                "gradient_steps": self.gradient_steps,
                "target_update_interval": self.target_update_interval,
                "exploration_initial_eps": self.exploration_initial_eps,
                "exploration_fraction": self.exploration_fraction,
                "exploration_final_eps": self.exploration_final_eps,
                "max_grad_norm": self.max_grad_norm,
            }
            
        return params


class ContiWorldLearningParams(GridWorldLearningParams):
    def __init__(
        self,
        algo_name: str = "dqn", 
        policy_kwargs: Dict[str, List[int]] = {"net_arch": [256, 256]}, # first 4 is embedding layer
        learning_starts: int = 1,
        train_freq: int = 1,
        gradient_steps: int = 1,
        target_update_interval: int = 1,
        buffer_size: int = 50000,
        batch_size: int = 32,
        max_grad_norm: float = 1, 
        # exploration related
        exploration_fraction: float = 0.8,
        exploration_initial_eps: float = 0.2,
        exploration_final_eps: float = 0.1,
        # learning related
        learning_rate_start: float = 1e-1,
        learning_rate_end: float = 1e-5,
        learning_fraction: float = 0.4, 
        # reward function related
        theta = "dist", 
        use_adrs: List[bool] = [False,],
        reward_types: list = ["progress", "hybrid", "distance"],
        adrs_update: int = 1000,
        adrs_mu: float = 0.5, 
        hybrid_eta: float = 0.001,
        version: int = 1, 
        # env related
        episode_step: int = int(1e3),
        env_name: str = "water",
        gamma: float = 0.9,
        vector: bool = False,
        node_embedding: bool = False,
        use_one_hot: bool = False,
        use_noise: bool = False,
        noise_level: float = 0,
        map_size: int = 1,
        map_id: int = 1,
        violation_end: bool = False,
        missing: bool = False,
        human: bool = False,
        # training related
        total_timesteps: int = int(1e4),
        total_run: int = 6,
        seed: int = 0,
        # evaluation related
        eval_freq: int = 100,
        rolling: int = 20,
        init_qs: dict = {"distance": 2, "hybrid": 2, "progress": 2},
        save: bool = True,
        cf: bool = False,
        # on-policy method
        n_steps: int = 2048,
        ent_coef: float = 1e-3,
        tau: float = 5e-3, 
    ):
        super().__init__(
            algo_name=algo_name, 
            policy_kwargs=policy_kwargs,
            learning_starts=learning_starts,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            target_update_interval=target_update_interval,
            buffer_size=buffer_size,
            batch_size=batch_size,
            max_grad_norm=max_grad_norm, 
            # exploration related
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            # learning related
            learning_rate_start=learning_rate_start,
            learning_rate_end=learning_rate_end,
            learning_fraction=learning_fraction, 
            # reward function related
            use_adrs=use_adrs,
            reward_types=reward_types,
            adrs_update=adrs_update,
            adrs_mu=adrs_mu,
            hybrid_eta=hybrid_eta,
            theta=theta, 
            version=version, 
            # env related
            episode_step=episode_step,
            env_name=env_name,
            gamma=gamma,
            vector=vector,
            node_embedding=node_embedding,
            use_one_hot=use_one_hot,
            use_noise=use_noise,
            noise_level=noise_level,
            map_size=map_size,
            map_id=map_id,
            violation_end=violation_end,
            missing=missing,
            human=human,
            # training related
            total_timesteps=total_timesteps,
            total_run=total_run,
            seed=seed,
            # evaluation related
            eval_freq=eval_freq,
            rolling=rolling,
            init_qs=init_qs,
            save=save,
            cf=cf
        )
        # map related
        self.water_world_map_path = "./psltl/envs/common/cont/water/maps/world.pkl"
        self.n_steps = n_steps
        self.tau = tau
        self.ent_coef = ent_coef
    
    def print_infos(self):
        print("=" * 75)
        print("Env: ", self.env_name)
        if self.env_name == "water":
            print("Map id:", self.map_id)
        print("Learning rate: ", self.learning_rate)
        print("Exploration Final Rate: ", self.exploration_final_eps)
        print("Gamma: ", self.gamma)
        print("Total Timesteps: ", self.total_timesteps)
        print("Total Runs: ", self.total_run)
        print("Reward Types: ", self.reward_types)
        print("Usage of Adaptive Reward Shaping: ", self.use_adrs)
        if True in self.use_adrs:
            print("Adaptive Reward Shaping Delta: ", self.hybrid_eta)
            print("Adaptive Reward Shaping Update Freqeuncy: ", self.adrs_update)
        print("Epsiode End Step: ", self.episode_step)

        print("-"*32 + "Model Params" + '-'*32)
        print("Buffer Size: ", self.buffer_size)
        print("Train Frequency: ", self.train_freq)
        print("Batch Size: ", self.batch_size)
        print("Final Exploration Rate: ", self.exploration_final_eps)
        print("Policy Net Architecture: ", self.policy_kwargs)

        print("=" * 75)

    def model_params(self, algo_name):
        assert algo_name in ["ddqn", "ddpg", "ppo", "td3", "sac", "a2c", "dqn"]
        common_params = {
            "policy_kwargs": self.policy_kwargs,
            "tensorboard_log": None,
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            }
        params = common_params.copy()
        if algo_name in ["ddqn", "dqn"]:
            params.update(
            {
                "double_dqn": False,
                "train_freq": self.train_freq,
                "batch_size": self.batch_size,
                "buffer_size": self.buffer_size,
                "learning_starts": self.learning_starts,
                "gradient_steps": self.gradient_steps,
                "target_update_interval": self.target_update_interval,
                "exploration_initial_eps": self.exploration_initial_eps,
                "exploration_fraction": self.exploration_fraction,
                "exploration_final_eps": self.exploration_final_eps,
                "max_grad_norm": self.max_grad_norm,
            }
            )
            if algo_name == "ddqn":
                params.update({"double_dqn": True})
        elif algo_name in ["td3", "sac", "ddpg"]:
            params.update({"tau": self.tau,
                           "batch_size": self.batch_size,
                           "buffer_size": self.buffer_size
                           })
        # On-Policy Algorithms
        elif algo_name in ["ppo"]:
            params.update({"batch_size": self.batch_size, "n_steps": self.n_steps, "ent_coef": self.ent_coef, "max_grad_norm": self.max_grad_norm,})
        elif algo_name in ["a2c"]:
            params.update({"n_steps": self.n_steps, "ent_coef": self.ent_coef, "max_grad_norm": self.max_grad_norm,})
        else:
            NotImplementedError(
                "model name is not implemented, we only support ddqn, ddpg, ppo, td3, sac, a2c currently.")
        return params