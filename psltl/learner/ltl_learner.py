import os
import numpy as np
from torch import nn
from typing import Any

# common
from psltl.envs.common.cont.water.water_world import WaterWorld, WaterWorldParams

# LTL environments
from psltl.envs.ltl_envs.grids.ltl_tax_env import LTLTaxiEnv
from psltl.envs.ltl_envs.grids.ltl_toy_env import LTLToyEnv
from psltl.envs.ltl_envs.grids.ltl_office_env import LTLOfficeEnv
from psltl.envs.ltl_envs.cont.ltl_water_env import LTLWaterEnv
from psltl.envs.ltl_envs.cont.ltl_cheetah_env import MyHalfCheetahEnv, LTLCheetahEnv

# save model param info
from psltl.utils.utils import save_model_param_info, set_seed

# libaries for algorithm, tabular-q to NN based algo
from psltl.rl_agents.common.callbacks import EvalCallback
from psltl.rl_agents.dqn.dqn import DQN
from psltl.rl_agents.dqn.policies import DQNPolicy
from psltl.ltl.ltl_utils import get_atm

# from stable baseline3
from stable_baselines3 import A2C, PPO, DDPG, TD3, SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecMonitor
from stable_baselines3.common.logger import configure


def get_ltl_env(
    env_name: str,
    reward_kwargs: dict,
    setting: dict,
    params: Any, 
    ) -> Any:
    
    # hyperparemters for environments
    max_episode_steps = params.episode_step

    # automaton setup
    atm = get_atm(env_name)
    atm.print_results()
    set_seed(params.seed)
    reward_kwargs.update({"version": params.version})

    if env_name == "office":
        map_size = params.map_size
        env = LTLOfficeEnv(atm, start=(2, 1), map_size=map_size, max_episode_steps=max_episode_steps, reward_kwargs=reward_kwargs, setting=setting)
        # for evaluation env
        reward_kwargs.update({"reward_type": "naive", "adaptive_rs": False})
        eval_env = LTLOfficeEnv(atm, start=(2, 1), map_size=map_size, max_episode_steps=max_episode_steps, reward_kwargs=reward_kwargs, setting=setting)
    elif env_name == "taxi":
        env = LTLTaxiEnv(atm, max_episode_steps, reward_kwargs=reward_kwargs, setting=setting)
        # for evaluation env
        reward_kwargs.update({"reward_type": "naive", "adaptive_rs": False})
        eval_env = LTLTaxiEnv(atm, max_episode_steps, reward_kwargs=reward_kwargs, setting=setting)
    elif env_name == "toy":
        env = LTLToyEnv(atm, reward_kwargs=reward_kwargs, setting=setting)
        # for evaluation env
        reward_kwargs.update({"reward_type": "naive", "adaptive_rs": False})
        eval_env = LTLToyEnv(atm, reward_kwargs=reward_kwargs, setting=setting)
    # For the continuous state space, so we use NN for the following environments
    elif env_name == "water":
        water_params = WaterWorldParams(params.water_world_map_path, b_radius=15, max_x=400, max_y=400, b_num_per_color=2, use_velocities=True, ball_disappear=False)
        water_env = WaterWorld(water_params)
        env = LTLWaterEnv(water_env, atm, max_episode_steps, reward_kwargs, setting)
        env.action_space.seed(params.seed)
        # for evaluation env
        reward_kwargs.update({"reward_type": "naive", "adaptive_rs": False})
        eval_env = LTLWaterEnv(water_env, atm, max_episode_steps, reward_kwargs, setting)
        eval_env.action_space.seed(params.seed)
    elif env_name == "cheetah":
        cheetah_env = MyHalfCheetahEnv()
        raw_env = LTLCheetahEnv(cheetah_env, atm, max_episode_steps=1000, reward_kwargs=reward_kwargs, setting=setting)
        raw_env.action_space.seed(params.seed)
        env = DummyVecEnv([lambda: raw_env])
        env = VecNormalize(env, norm_reward=False)
        # for evaluation env
        reward_kwargs.update({"reward_type": "naive", "adaptive_rs": False})
        cheetah_env = MyHalfCheetahEnv()
        eval_raw_env = LTLCheetahEnv(cheetah_env, atm, max_episode_steps=1000, reward_kwargs=reward_kwargs, setting=setting)
        eval_raw_env.action_space.seed(params.seed)
        eval_env = DummyVecEnv([lambda: eval_raw_env])
        eval_env = VecNormalize(eval_env, norm_reward=False)
        eval_env = VecMonitor(eval_env)
    else:
        ValueError("not implemented yet")

    return env, eval_env


def ltl_env_learn(
    reward_type: str,
    env: Any,
    eval_env: Any,
    params: Any,
    ):

    env_name = params.env_name
    algo_name = params.algo_name
    total_timesteps = params.total_timesteps
    seed = params.seed
    init_q = params.init_qs
    model_params = params.model_params(algo_name)
    
    model_params.update({"verbose": 0, "seed": seed})
    eval_freq = params.eval_freq
    folder_name = env_name

    if float(params.noise_level) > 0:
        folder_name += "_noise"
    if params.missing:
        folder_name += "_infeasible"
    if params.use_adrs:
        reward_type += "_adrs"

    log_path = "./log/" + folder_name + "/" + algo_name + "/" + reward_type + f"_theta{params.theta}_update{params.adrs_update}" + "/" + str(seed)
    eval_callback = EvalCallback(eval_env, eval_freq=eval_freq, log_path=log_path, discount=params.gamma, eval_window=int(params.rolling))
    
    # grid world environments
    if env_name in ["taxi", "office", "toy"]:
        model = DQN(DQNPolicy, env, **model_params)
        sequential_model_container = model.q_net.q_net
        layer = sequential_model_container[0]
        if "progress" in reward_type:
            layer.weight = nn.init.constant_(layer.weight, init_q["progress"])
        elif "hybrid" in reward_type:
            layer.weight = nn.init.constant_(layer.weight, init_q["hybrid"])
        elif "naive" in reward_type:
            layer.weight = nn.init.constant_(layer.weight, init_q["naive"])

    elif env_name in ["cheetah", "water"]:
        if algo_name == "ddpg" or algo_name == "td3":
            assert len(env.action_space.shape) > 0, "DDPG and TD3 need continuous action space"
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
            
        if algo_name == "ddpg":
            model = DDPG("MlpPolicy", env, action_noise=action_noise, **model_params)
        
        elif algo_name == "td3":
            model = TD3("MlpPolicy", env, action_noise=action_noise, **model_params)

        elif algo_name == "ddqn" or algo_name == "dqn":
            model = DQN("MlpPolicy", env, **model_params)

        elif algo_name == "a2c":
            model = A2C("MlpPolicy", env, **model_params)

        elif algo_name == "ppo":
            model = PPO("MlpPolicy", env, **model_params)

        elif algo_name == "sac":
            model = SAC("MlpPolicy", env, **model_params)

    # training the algorithm for total timesteps
    logger = configure(os.getcwd() + "/logger/" + folder_name + "/" + algo_name + "/" + reward_type + f"_theta{params.theta}_update{params.adrs_update}" + "/" + str(seed), ["csv"])
    set_seed(params.seed)
    model.set_logger(logger)
    model.learn(total_timesteps=total_timesteps, callback=eval_callback, log_interval=1000)
    # save_model_param_info(model, log_path, algo_name, params=params)