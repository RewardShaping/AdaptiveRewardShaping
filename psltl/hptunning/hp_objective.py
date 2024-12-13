from typing import Any
import os
import pickle5 as pickle
import gym
import optuna
import torch as th
import numpy as np
from torch import nn
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from psltl.rl_agents.common.callbacks import EvalCallback
# from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.env_util import DummyVecEnv
from psltl.envs.skeletons.env_default_settings import setting, reward_kwargs
from psltl.learner.ltl_learner import get_ltl_env
from psltl.hptunning.param_setting import POLICIES, ALGORITMHS, sample_a2c_params, sample_dqn_params, sample_ddpg_params, sample_ddqn_params, THETA_SIZE, ADRS_UPDATE


class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 50000,
        deterministic: bool = True,
        verbose: int = 0,
        discount: float = 1.,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
            discount=discount
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            eval_metric = (self.last_mean_reward + self.recent_success_rate) / 2
            # eval_metric = self.last_mean_reward
            self.trial.report(eval_metric, self.eval_idx)
            # Prune trial if need.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
            
        return True


def objective(
    trial: optuna.Trial,
    N_EVAL_EPISODES: int,
    EVAL_FREQ: int,
    N_TIMESTEPS: int,
    learning_params: Any,
    ) -> float:

    # hybrid_eta = trial.suggest_float("hybrid_eta", 0.00000001, 0.1, log=True)
    # adrs_update = trial.suggest_int("adrs_update", int(0.1 * N_TIMESTEPS), N_TIMESTEPS, step=int(0.1 * N_TIMESTEPS))
    env_name = learning_params.env_name
    algo_name = learning_params.algo_name
    
    NN_SIZE = dict([
    ("small", [256, 256]),
    ("middle", [256, 256, 256]),
    ("big", [512, 512]),
    ("huge", [1024, 1024, 1024]),
    ])
    
    hybrid_eta = 1e-2
    adrs_update = trial.suggest_categorical("adrs_update", list(ADRS_UPDATE.keys()))
    adrs_update = ADRS_UPDATE[adrs_update]
    theta_size = trial.suggest_categorical("theta", list(THETA_SIZE.keys()))
    theta = THETA_SIZE[theta_size]
    reward_kwargs.update(dict([
        ("hybrid_eta", hybrid_eta), 
        ("adrs_update", adrs_update), 
        ("adrs_mu", 0.9), 
        ("theta", theta),
        ("reward_type", learning_params.reward_types),
        ]))
    
    # use_ards = [False] or [True, False]
    if bool(learning_params.use_adrs):
        reward_kwargs.update({"adaptive_rs": True})

    kwargs = {}

    # we will mainly test batch size, buffer size, learning start, learning end, learning fraction, exploration rate, exploration ,,,
    param_ft = {"dqn": sample_dqn_params, "a2c": sample_a2c_params, "ppo": sample_a2c_params, 
                "ddqn": sample_ddqn_params, "ddpg": sample_ddpg_params
                }
    kwargs.update(param_ft[algo_name](trial))
    # Add Policy information
    kwargs.update({"policy": POLICIES[algo_name]})
    
    # Sample DQN hyperparams
    
    # kwargs.update({"batch_size": 8, "buffer_size": 8})
    setting.update(dict([(
        "vector", learning_params.vector), 
        ("use_one_hot", learning_params.use_one_hot), 
        ("adrs_update", learning_params.adrs_update), 
        ("node_embedding", learning_params.node_embedding),
        ("missing", bool(learning_params.missing)), 
        ("human", learning_params.human), 
        ("noise", learning_params.noise_level)
            ]))
    discount = 1
    
    # net arch will be updated here
    if env_name == "taxi":
        learning_rate_start = trial.suggest_float("learning_rate_start", 0.5, 1, step=0.1)
        learning_rate_end = trial.suggest_float("learning_rate_end", 0.001, 0.5, log=True)
        learning_rate_fraction = trial.suggest_float("learning_fraction", 0.1, 1, step=0.1)
        # target_update_interval = trial.suggest_int("target_update_interval", 0, 2000, step=100)
        target_update_interval = 1

        policy_kwargs = dict([
                    ("with_bias", False), 
                    ("net_arch", []), 
                    ("optimizer_class", th.optim.SGD),
                        ])
        
        kwargs["learning_starts"] = 0
        kwargs["gradient_steps"] = 1
        kwargs["max_grad_norm"] = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)

        setting.update({"vector": True})
        
        kwargs.update({
        "exploration_initial_eps": 1., "exploration_final_eps": 0.1, "exploration_fraction": 0.2,
        "learning_rate_start": learning_rate_start, "learning_rate_end": learning_rate_end, "target_update_interval": target_update_interval,
        "learning_rate_fraction": learning_rate_fraction
        })
    
    elif env_name == "office":
        # learning_rate_start = trial.suggest_float("learning_rate_start", 0.5, 1, step=0.1)
        learning_rate_end = trial.suggest_float("learning_rate_end", 0.1, 0.5, step=0.1)
        learning_rate_fraction = trial.suggest_float("learning_fraction", 0.1, 1, step=0.1)
        exploration_initial_eps = trial.suggest_float("exploration_initial_eps", 0.1, 1, step=0.1)
        exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 1, step=0.1)
        gradient_steps = trial.suggest_int("gradinet_steps", 1, 4, step=1)
        policy_kwargs = {"net_arch": [], "with_bias": False, "optimizer_class": th.optim.SGD}

        
        learning_rate_start = 1.
        learning_rate_fraction = 1.
        
        kwargs.update({
            "exploration_initial_eps": exploration_initial_eps,
            "exploration_final_eps": 0.1,
            "exploration_fraction": exploration_fraction,
            "learning_rate_start": learning_rate_start, 
            "learning_rate_end": learning_rate_end, 
            "learning_rate_fraction": learning_rate_fraction,
            "gradient_steps": gradient_steps,
            "max_grad_norm": 1,
            "learning_starts": 0, 
            "target_update_interval": 1
            })
        discount = 0.95
    
    elif env_name == "water":
        # nn_size = trial.suggest_categorical("nn_size", list(NN_SIZE.keys()))
        # nn_size = NN_SIZE[nn_size]
        nn_size = [256, 256]
        learning_rate_start = trial.suggest_float("learning_rate_start", 1e-5, 0.8, log=True)
        learning_rate_end = trial.suggest_float("learning_rate_end", 1e-7, 0.01, log=True)
        learning_rate_fraction = trial.suggest_float("learning_fraction", 0.1, 1, step=0.1)
        target_update_interval = trial.suggest_int("target_update_interval", 0, 2000, step=100)
        # exploration_final_eps = trial.suggest_float("exploration final", 0.01, 0.1, log=True)
        target_update_interval = max(target_update_interval, 1)
        policy_kwargs = {"net_arch": nn_size}
        kwargs["gradient_steps"] = 1
        kwargs["max_grad_norm"] = 1
        kwargs["learning_starts"] = 1000
        setting.update({"vector": False})
        kwargs.update({
        "exploration_initial_eps": 1., "exploration_final_eps": 0.1, "exploration_fraction": 0.2,
        "learning_rate_start": learning_rate_start, "learning_rate_end": learning_rate_end, "target_update_interval": target_update_interval,
        "learning_rate_fraction": learning_rate_fraction,
        })
        discount = 0.9

    elif env_name == "cheetah":
        # nn_size = trial.suggest_categorical("nn_size", list(NN_SIZE.keys()))
        # nn_size = NN_SIZE[nn_size]
        nn_size = [256, 256]
        learning_rate_start = trial.suggest_float("learning_rate_start", 1e-5, 0.8, log=True)
        learning_rate_end = trial.suggest_float("learning_rate_end", 1e-7, 0.01, log=True)
        learning_rate_fraction = trial.suggest_float("learning_fraction", 0.1, 1, step=0.1)
        policy_kwargs = {"net_arch": nn_size}
        kwargs["learning_starts"] = 0
        kwargs.update({
        "learning_rate_start": learning_rate_start, "learning_rate_end": learning_rate_end, "learning_rate_fraction": learning_rate_fraction,
        })
        discount = 0.99

    env, eval_env = get_ltl_env(env_name, reward_kwargs, setting, learning_params)
    kwargs["policy_kwargs"] = policy_kwargs
    
    learning_rate = get_linear_fn(kwargs["learning_rate_start"], kwargs["learning_rate_end"], kwargs["learning_rate_fraction"])
    kwargs.update({"learning_rate": learning_rate})
    kwargs.pop("learning_rate_start")
    kwargs.pop("learning_rate_end")
    kwargs.pop("learning_rate_fraction")
    kwargs.update({"env": env})
    # Create the RL model.
    model = ALGORITMHS[algo_name](**kwargs)
    if env_name == "office":
        sequential_model_container = model.q_net.q_net
        layer = sequential_model_container[0]
        layer.weight = nn.init.constant_(layer.weight, 2)
    elif env_name == "taxi":
        sequential_model_container = model.q_net.q_net
        layer = sequential_model_container[0]
        layer.weight = nn.init.constant_(layer.weight, 4)
    if env_name == "cheetah":
        pass
    else:
        # Create env used for evaluation.
        eval_env = Monitor(eval_env)
    # Create the callback that will periodically evaluate and report the performance.
    eval_callback = TrialEvalCallback(
        eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=True, discount=discount
    )

    nan_encountered = False
    try:
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN.
        print(e)
        nan_encountered = True
    finally:
        # Free memory.
        model.env.close()
        eval_env.close()

    # Tell the optimizer that the trial failed.
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()
    
    # optimize with considering with mean reward and success rate
    env2maxr = {"office": 1.19, "taxi": 1.3, "water": 0.27, "cheetah": 5.}
    return (eval_callback.last_mean_reward / env2maxr[env_name] + eval_callback.recent_success_rate) / 2
    # return eval_callback.last_mean_reward


def hp_tunning(args, learning_params):
    algo_name = args.algo_name
    env_name = args.env_name
    folder_name = env_name
    reward_type = learning_params.reward_types

    if float(learning_params.noise_level) > 0:
        folder_name += "_noise"
    if learning_params.missing:
        folder_name += "_infeasible"
    if learning_params.use_adrs:
        reward_type += "_adrs"

    save_path = "./best_hp/" + folder_name + "/" + algo_name + "/" + reward_type + "/" + str(learning_params.seed)
    env2eval_freq = {"office": 10_000, "taxi": 100_000, "water": 250_000, "cheetah": 250_000}
    env2total_steps = {"office": 60_000, "taxi": 500_000, "water": 2_000_000, "cheetah": 2_000_000}
    env2trials = {"taxi": 1_000, "water": 100, "office": 2000, "cheetah": 100}
    
    # a trial typically corresponds to a set of hyperparameters that are evaluated to find the optimal configuration.
    N_TRIALS = env2trials[env_name]
    # the number of initial trials used for exploration before the optimization algorithm starts focusing on exploitation. These initial trials are often used to explore a diverse set of hyperparameter configurations.
    N_STARTUP_TRIALS = 10
    # It is the overall budget for evaluating different sets of hyperparameters.
    N_EVALUATIONS = 10
    N_EVAL_EPISODES = 10
    N_TIMESTEPS = env2total_steps[env_name]
    EVAL_FREQ = env2eval_freq[env_name]

    # Set pytorch num threads to 1 for faster training.
    th.set_num_threads(1)
    
    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used.
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    
    wrapped_objective = lambda x: objective(x, N_EVAL_EPISODES, EVAL_FREQ, N_TIMESTEPS, learning_params)
    try:
        study.optimize(wrapped_objective, n_trials=N_TRIALS, timeout=600)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    trial.params["value"] = trial.value
    
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(save_path + "/best_params.pkl", 'wb') as f:
        pickle.dump(trial.params, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    algo_name = "ddqn"
    kwargs = {}
    # Add Policy information
    kwargs.update({"policy": POLICIES[algo_name]})
