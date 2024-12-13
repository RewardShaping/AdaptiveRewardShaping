import os
import random
import numpy as np
import torch as th


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def save_param_info(folder_path, algo_name, params):
    # Parameter information
    reward_types = params.reward_types
    use_adrs = params.use_adrs
    # set up environment
    ep_step = params.episode_step
    hybrid_eta = params.hybrid_eta
    adrs_mu = params.adrs_mu
    adrs_update = params.adrs_update
    gamma = params.gamma
    total_timesteps = params.total_timesteps
    exploration_initial_eps = params.exploration_initial_eps,
    exploration_fraction = params.exploration_fraction,
    exploration_final_eps = params.exploration_final_eps,
    map_id = params.map_id
    env_name = params.env_name
    lr_start = params.learning_rate_start
    lr_end = params.learning_rate_end
    lr_fraction = params.learning_fraction

    model_params = params.model_params(algo_name)

    with open(folder_path + "/" + "model param info.txt", "w") as f:
        f.write("env name: " + str(env_name) + "\n")
        f.write("map id: " + str(map_id) + "\n")
        f.write("algorithm name: " + algo_name + "\n")
        f.write("reward types: " + str(reward_types) + "\n")
        f.write("use adrs: " + str(use_adrs) + "\n")
        f.write("episode step: " + str(ep_step) + "\n")
        f.write("adrs mu: " + str(adrs_mu) + "\n")
        f.write("hybrid eta: " + str(hybrid_eta) + "\n")
        f.write("adrs update: " + str(adrs_update) + "\n")
        f.write("gamma: " + str(gamma) + "\n")
        f.write("total_timesteps: " + str(total_timesteps) + "\n")
        f.write("exploration_final_eps: " + str(exploration_final_eps) + "\n")
        f.write("exploration_initial_eps: " + str(exploration_initial_eps) + "\n")
        f.write("exploration_fraction: " + str(exploration_fraction) + "\n")
        f.write("learning rate start: " + str(lr_start) + "\n")
        f.write("learning_rate_end: " + str(lr_end) + "\n")
        f.write("learning_rate_fraction: " + str(lr_fraction) + "\n")
        f.write("seed: " + str(params.seed) + "\n")
        f.write("noise_level: " + str(params.noise_level) + "\n") 
    
        for item in model_params.items():
            f.write(str(item))


def save_model_param_info(model, folder_path, algo_name, params):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    save_path = folder_path + "/" + algo_name
    model.save(save_path)

    save_param_info(folder_path, algo_name, params)
