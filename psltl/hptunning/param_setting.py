""" Optuna example that optimizes the hyperparameters of
a reinforcement learning agent using A2C implementation from Stable-Baselines3
on an OpenAI Gym environment.
This is a simplified version of what can be found in https://github.com/DLR-RM/rl-baselines3-zoo.
You can run this example as follows:
    $ python sb3_simple.py
"""
from typing import Any
from typing import Dict

import gym
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from psltl.rl_agents.dqn.dqn import DQN
from psltl.rl_agents.dqn.policies import DQNPolicy

from stable_baselines3 import DDPG, PPO, A2C
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

import torch as th
import torch.nn as nn


ALGORITMHS = {"dqn": DQN, "ddqn": DQN, "ddpg": DDPG, "a2c": A2C, "ppo": PPO}

POLICIES = {"dqn": DQNPolicy, "ddqn": DQNPolicy, "a2c": "MlpPolicy", "ppo": "MlpPolicy", "ddpg": "MlpPolicy"}

BIG_BATCH_MAP = dict([
    ("normal", 32), 
    ("tiny", 64),
    ("small", 128),
    ("middle", 256),
    ("huge", 512),
])

# BUFFER_SIZE = dict([
#     ("online", 1),
#     ("normal", 32),
#     ("tiny", 64),
#     ("small", 128),
#     ("middle", 256),
#     ("huge", 512),
#     ("big", 50000),
# ])

BATCH_MAP = dict([
    ("online", 2),
    ("normal", 3), 
    ("tiny", 4),
    ("small", 8),
    ("middle", 16),
    ("huge", 32),
])

BUFFER_SIZE = dict([
    ("online", 2),
    ("normal", 3),
    ("tiny", 4),
    ("small", 8),
    ("middle", 16),
    ("huge", 32),
    ("big", 64),
])

EXP_FINAL = dict([
    ("five", 0.05),
    ("ten", 0.1)
    ])

EXP_FRACTION = dict([
    ("one", 0.1),
    ("two", 0.2),
    ("three", 0.3), 
    ("four", 0.4),
    ("five", 0.5),
    ("six", 0.6)
])

THETA_SIZE = dict([
        ("small", 100), 
        ("middle", 200),
        ("big", 500),
        ("huge", 1000)
    ])

ADRS_UPDATE = dict([
    ("small", 1000), 
    ("middle", 2000),
    ("big", 4000),
    ("huge", 5000)
])

def sample_common_params(trial: optuna.Trial) -> Dict[str, Any]:
    gamma = 1.0 - trial.suggest_float("gamma", 0.05, 0.1, log=True)
    learning_rate_start = trial.suggest_float("learning_rate_start", 1e-5, 1., log=True)
    learning_rate_end = trial.suggest_float("learning_rate_end", 1e-7, 1., log=True)

    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("learning_rate_start_", learning_rate_start)
    trial.set_user_attr("learning_rate_end_", learning_rate_end)
    
    return {
        "gamma": gamma,
        "learning_rate_start": learning_rate_start,
        "learning_rate_end": learning_rate_end,
    }

def sample_ddqn_params(trial: optuna.Trial) -> Dict[str, Any]:
    batch_size = trial.suggest_categorical("batch_size", list(BIG_BATCH_MAP.keys()))
    batch_size = BIG_BATCH_MAP[batch_size]

    buffer_size = 50_000

    return {
        "batch_size": batch_size, 
        "buffer_size": buffer_size,
        "double_dqn": True,
    }


def sample_dqn_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for DQN hyperparameters."""
    
    batch_size = trial.suggest_categorical("batch_size", list(BATCH_MAP.keys()))
    batch_size = BATCH_MAP[batch_size]
    buffer_size = batch_size

    # buffer_size = trial.suggest_categorical("buffer_size", list(BUFFER_SIZE.keys()))
    # buffer_size = BUFFER_SIZE[buffer_size]

    # exp_fraction = trial.suggest_categorical("exploration_fraction", list(EXP_FRACTION.keys()))
    # exploration_fraction = EXP_FRACTION[exp_fraction]

    # exp_final = trial.suggest_categorical("exploration_final_eps", list(EXP_FINAL.keys()))
    # exploration_final_eps = EXP_FINAL[exp_final]

    # exploration_initial_eps = trial.suggest_float("exploration_initial_eps", 0.1, 1., log=True)
    # trial.set_user_attr("exploration_initial_eps", exploration_initial_eps)
    # trial.set_user_attr("exploration_final_eps_", exploration_final_eps)
    # trial.set_user_attr("exploration_fraction_", exploration_fraction)

    # exploration_initial_eps = 0.2
    # exploration_final_eps = 0.1
    # exploration_fraction = 0.7

    return {
        "batch_size": batch_size, 
        "buffer_size": buffer_size,
    }


def sample_a2c_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for DQN hyperparameters."""
    
    net_arch = trial.suggest_categorical("net_arch", ["tiny"])
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    ortho_init = trial.suggest_categorical("ortho_init", [False, True])

    net_arch = [
        {"pi": [4, 64, 64], "vf": [4, 64, 64]} if net_arch == "tiny" else {"pi": [64, 64, 64], "vf": [64, 64, 64]}
    ]

    return {
        "ent_coef": ent_coef,
        "policy_kwargs": {
            "net_arch": net_arch,
            "ortho_init": ortho_init,
            "activation_fn": nn.ReLU,
            "optimizer_class": th.optim.Adam
        },
    }

def sample_ddpg_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {}