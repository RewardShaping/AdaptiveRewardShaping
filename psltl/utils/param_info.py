from psltl.learner.learning_param import GridWorldLearningParams, ContiWorldLearningParams
import torch as th

reward_types = ["progress", "hybrid", "naive"]

####################################################################
# HYPERPARAMS FOR TOY ENVIRONMENT
####################################################################

toy = {
    "policy_kwargs": dict([("with_bias", False), ("net_arch", []), ("optimizer_class", th.optim.SGD),]),
    "algo_name": "dqn",
    "gamma": 0.9,
    "exploration_initial_eps": 0.2,
    "exploration_fraction": 1,
    "exploration_final_eps": 0.2,
    "learning_rate_start": 1.,
    "learning_rate_end": 1., 
    "learning_fraction": 0.7, 
    "learning_starts": 0,
    "gradient_steps": 1,
    "train_freq": 1, 
    "adrs_update": 10, 
    "batch_size": 1,
    "buffer_size": 1,
    "max_grad_norm": 1, 
    "target_update_interval": 1,
    "hybrid_eta": 0.1,
    "episode_step": 25, 
    "init_qs": {"progress": 2, "distance": 2, "hybrid": 2, "naive": 2},
    "node_embedding": True,
    }

toy = dict([(reward_type, toy) for reward_type in reward_types])

####################################################################
# HYPERPARAMS FOR OFFICE ENVIRONMENT
####################################################################

office_progress = {
    "algo_name": "dqn",
    "exploration_initial_eps": 0.2, # 0.2
    "exploration_fraction": 1, #1 
    "exploration_final_eps": 0.1, # 0.1
    "learning_rate_start": 1., # 1.
    "learning_rate_end": 0.1, # 0.5
    "learning_fraction": 1., # 0.5 
    "learning_starts": 0,
    "gradient_steps": 3,
    "train_freq": 1, 
    "batch_size": 16,
    "buffer_size": 16,
    "hybrid_eta": 0.01,
    "max_grad_norm": 1, 
    "target_update_interval": 1,
    "init_qs": {"progress": 2, "distance": 2, "hybrid": 2, "naive": 2}, # 2 was good for 
    "policy_kwargs": dict([("with_bias", False), ("net_arch", []), ("optimizer_class", th.optim.SGD),]),
    "gamma": 0.95,
    "theta": 100, # 100 
    "adrs_update": 500, # 500
    "node_embedding": True,
    "eval_freq": 100,
    "total_timesteps": 60_000
    }

office_hybrid = office_progress.copy()
office_hybrid.update({
    "theta": 100,
    "adrs_update": 300,
    "batch_size": 8,
    "buffer_size": 8,
    "learning_rate_end": .1, 
    "learning_fraction": 1., 
    "gradient_steps": 4, 
    })

office_naive = office_progress.copy()
office_naive.update({
    "batch_size": 1,
    "buffer_size": 1,
    "learning_rate_end": .1, 
    "learning_fraction": 1.,
    "gradient_steps": 1, 
    "exploration_initial_eps": 0.2,
    "exploration_fraction": 1, #1 
    "exploration_final_eps": 0.1, # 0.1
    "theta": 0,
    })

office_missing_progress = office_progress.copy()
office_missing_progress.update({
    "theta": 100,
    "adrs_update": 500,
    "batch_size": 2,
    "buffer_size": 2,
    "learning_rate_end": .5, 
    "learning_fraction": .8, 
    "gradient_steps": 2, 
    })

office_missing_hybrid = office_hybrid.copy()
office_missing_hybrid.update({
    "theta": 100,
    "adrs_update": 300,
    "batch_size": 16,
    "buffer_size": 16,
    "learning_rate_end": .1, 
    "learning_fraction": 1., 
    "gradient_steps": 3, 
    })

office_missing_naive = office_naive.copy()
office_missing_naive.update({
    "theta": 0,
    "batch_size": 1,
    "buffer_size": 1,
    "learning_rate_end": .1, 
    "learning_fraction": 1., 
    "gradient_steps": 1, 
    })

office_noise_progress = office_progress.copy()
office_noise_progress.update({
    "theta": 100,
    "adrs_update": 1000,
    "batch_size": 512,
    "buffer_size": 512,
    "learning_rate_end": .3, 
    "learning_fraction": .7,
    "gradient_steps": 3, 
    "exploration_initial_eps": 0.7,
    "exploration_fraction": 0.4,
    })

office_noise_hybrid = office_hybrid.copy()
office_noise_hybrid.update({
    "theta": 200,
    "adrs_update": 1000,
    "batch_size": 64,
    "buffer_size": 64,
    "learning_rate_end": .2, 
    "learning_fraction": .3,
    "gradient_steps": 1, 
    "exploration_initial_eps": 0.4,
    "exploration_fraction": 0.6,
    })

office_noise_naive = office_naive.copy()
office_noise_naive.update({
    "theta": 0,
    "batch_size": 512,
    "buffer_size": 512,
    "learning_rate_end": .3, 
    "learning_fraction": .6,
    "gradient_steps": 1, 
    "exploration_initial_eps": 0.4,
    "exploration_fraction": 0.4,
    })


office_normal = {"progress": office_progress, "hybrid": office_hybrid, "naive": office_naive}
office_missing = {"progress": office_missing_progress, "hybrid": office_missing_hybrid, "naive": office_missing_naive}
office_noise = {"progress": office_noise_progress, "hybrid": office_noise_hybrid, "naive": office_naive}

office_params = {}
office_params["normal"] = office_normal
office_params["noise"] = office_noise
office_params["missing"] = office_missing

####################################################################
# HYPERPARAMS FOR TAXI ENVIRONMENT
####################################################################

taxi_progress = {
    "algo_name": "dqn",
    "hybrid_eta": 0.005, # 0.005
    "init_qs": {"progress": 4, "distance": 4, "hybrid": 4, "naive": 4},
    "exploration_initial_eps": 1., # 
    "exploration_fraction": 0.2, # 0.2
    "exploration_final_eps": 0.1, # 0.1
    "learning_rate_start": 1.,
    "learning_rate_end": 0.5, # 0.5
    "learning_fraction": 1., # 1.
    "learning_starts": 0,
    "train_freq": 1, 
    "batch_size": 2,
    "buffer_size": 2,
    "target_update_interval": 1,
    "gamma": 0.9,
    "gradient_steps": 1,
    "policy_kwargs": dict([("with_bias", False), ("net_arch", []), ("optimizer_class", th.optim.SGD),]),
    "max_grad_norm": 1.,
    "theta": 100, # 50
    "adrs_update": 4000, # 4000
    "node_embedding": True,
    "eval_freq": 1000,
    "total_timesteps": 500_000
    }

taxi_naive = taxi_progress.copy()

taxi_hybrid = taxi_progress.copy()
taxi_hybrid.update(
    {
        "batch_size": 16, # 16
        "buffer_size": 16, # 16
        "learning_rate_end": 1e-5, # 0.01
        "learning_rate_start": 1., # 1.
        "learning_fraction": 0.5, # 0.5 
        "gradient_steps": 2,
        "hybrid_eta": 0.01, # 0.005
        "theta": 100, 
        "adrs_update": 5000,
    }
    )

taxi_missing_progress = taxi_progress.copy()
taxi_missing_progress.update({
    "batch_size": 4,
    "buffer_size": 4, # 16
    "theta": 100,
    "adrs_update": 1000,
    "learning_rate_start": 1.,
    "learning_rate_end": 0.1,
    "learning_fraction": .5, 
    "gradient_steps": 2,
    "exploration_fraction": 0.2, # 0.2
})

taxi_missing_hybrid = taxi_progress.copy()
taxi_missing_hybrid.update({
    "batch_size": 8,
    "buffer_size": 8, # 16
    "theta": 100,
    "adrs_update": 2000,
    "learning_rate_start": 1.,
    "learning_rate_end": 0.01,
    "learning_fraction": 1., 
    "gradient_steps": 2,
    "exploration_fraction": 0.2, # 0.2
})

taxi_missing_naive = taxi_naive.copy()
taxi_missing_naive.update(
    {
        "theta": 0,
        "batch_size": 4, # 16
        "buffer_size": 4, # 16
        "learning_rate_end": 0.1, # 0.01
        "learning_rate_start": 1., # 1.
        "learning_fraction": 1., # 0.5 
        "gradient_steps": 2,
        "exploration_fraction": 0.2, # 0.2
    })

taxi_noise_progress = taxi_progress.copy()
taxi_noise_progress.update({
    "batch_size": 512,
    "buffer_size": 512, # 16
    "theta": 1000,
    "adrs_update": 4000,
    "learning_rate_start": .8,
    "learning_rate_end": 0.4,
    "learning_fraction": 0.1, 
})

taxi_noise_hybrid = taxi_hybrid.copy()
taxi_noise_hybrid.update({
    "batch_size": 512,
    "buffer_size": 512, # 16
    "theta": 200,
    "adrs_update": 2000,
    "learning_rate_start": .8,
    "learning_rate_end": 0.001,
    "learning_fraction": 0.8, 
})

taxi_noise_naive = taxi_naive.copy()
taxi_noise_naive.update({
    "batch_size": 16,
    "buffer_size": 16, # 16
    "theta": 500,
    "adrs_update": 4000,
    "learning_rate_start": .8,
    "learning_rate_end": 0.05,
    "learning_fraction": 0.1, 
})


taxi_normal = {"progress": taxi_progress, "hybrid": taxi_hybrid, "naive": taxi_naive, "hltlhybrid": taxi_progress, "hltlnaive": taxi_progress}
taxi_missing = {"progress": taxi_missing_progress, "hybrid": taxi_missing_hybrid, "naive": taxi_naive}
taxi_noise = {"progress": taxi_noise_progress, "hybrid": taxi_noise_hybrid, "naive": taxi_naive}

taxi_params = {}
taxi_params["normal"] = taxi_normal
taxi_params["noise"] = taxi_noise
taxi_params["missing"] = taxi_missing

####################################################################
# HYPERPARAMS FOR WATER ENVIRONMENT
####################################################################

water_progress = {
    "policy_kwargs": dict([("net_arch", [256, 256]), ("optimizer_class", th.optim.Adam),]),
    "hybrid_eta": 0.001,
    "exploration_initial_eps": 1.,
    "exploration_fraction": 0.2,
    "exploration_final_eps": 0.1,
    "learning_rate_start": 1e-04,
    "learning_rate_end": 1e-05,
    "learning_fraction": 0.8, 
    "train_freq": 1, 
    "batch_size": 64,
    "buffer_size": int(5e4),
    "target_update_interval": 2000,
    "gamma": 0.9,
    "gradient_steps": 4,
    "learning_starts": 1000,
    "algo_name": "ddqn",
    "theta": 500,
    "adrs_update": 5000,
    "use_one_hot": True,
    "eval_freq": 1000,
    "total_timesteps": 2_000_000
    }

water_hybrid = water_progress.copy()
water_hybrid.update({"batch_size": 32})

water_naive = water_progress.copy()
water_naive.update({
    "batch_size": 512,
    "theta": 0,
    "adrs_update": 5000,
    "learning_rate_start": 1e-4,
    "learning_rate_end": 3e-5,
    "learning_fraction": 0.8, 
    "target_update_interval": 1500
})

water_noise_progress = water_progress.copy()
water_noise_progress.update({
    "batch_size": 128,
    "theta": 100,
    "adrs_update": 4000,
    "learning_rate_start": 1e-4,
    "learning_rate_end": 3e-5,
    "learning_fraction": 0.6, 
    "target_update_interval": 600
})

water_noise_hybrid = water_progress.copy()
water_noise_hybrid.update({
    "batch_size": 128,
    "theta": 1000,
    "adrs_update": 5000,
    "learning_rate_start": 1e-4,
    "learning_rate_end": 2e-7,
    "learning_fraction": 0.9, 
    "target_update_interval": 1500
})

water_noise_naive = water_progress.copy()
water_noise_naive.update({
    "batch_size": 64,
    "theta": 0,
    "adrs_update": 5000,
    "learning_rate_start": 1e-4,
    "learning_rate_end": 1e-6,
    "learning_fraction": 0.3, 
    "target_update_interval": 2000
})

water_normal = {"progress": water_progress, "hybrid": water_hybrid, "naive": water_progress}
water_noise = {"progress": water_noise_progress, "hybrid": water_noise_hybrid, "naive": water_noise_progress}
water_missing = water_normal.copy()

water_params = {}
water_params["noise"] = water_noise
water_params["normal"] = water_normal
water_params["missing"] = water_missing

####################################################################
# HYPERPARAMS FOR HALFCHEETAH ENVIRONMENT
####################################################################

cheetah_progress = {
    "policy_kwargs": {"net_arch": dict(pi=[300, 400], vf=[300, 400]), "optimizer_class": th.optim.RMSprop,},
    "hybrid_eta": 0.0005,
    "learning_rate_start": 3e-4,
    "learning_rate_end": 1e-5,
    "learning_fraction": 0.4, 
    "n_steps": 2048,
    "batch_size": 128,
    "gamma": 0.99,
    "ent_coef": 1e-3,
    "max_grad_norm": 0.1, 
    "algo_name": "ppo",
    "use_one_hot": True,
    "eval_freq": 1000,
    "total_timesteps": 2_000_000
    }

########################### PPO ###################################

cheetah_distance = cheetah_progress.copy()

cheetah_naive = cheetah_progress.copy()
cheetah_hybrid = cheetah_progress.copy()
cheetah_hybrid.update({"batch_size": 256})

cheetah_ppo = {"progress": cheetah_progress, "hybrid": cheetah_hybrid, "distance": cheetah_distance,\
           "naive": cheetah_naive, "success": cheetah_progress}

########################### A2C ###################################

cheetah_a2c_progress = cheetah_progress.copy()
cheetah_a2c_progress.update({"policy_kwargs": {"net_arch": dict(pi=[200, 300], vf=[200, 300]), "optimizer_class": th.optim.RMSprop,},
                             "learning_rate_start": 7e-4})

cheetah_a2c_distance = cheetah_a2c_progress.copy()

cheetah_a2c_naive = cheetah_a2c_progress.copy()

cheetah_a2c_hybrid = cheetah_a2c_progress.copy()
cheetah_a2c_hybrid.update({"batch_size": 256, "n_steps": 2048})

cheetah_a2c = {"progress": cheetah_a2c_progress, "hybrid": cheetah_a2c_hybrid, "distance": cheetah_a2c_distance,\
           "naive": cheetah_a2c_naive, "success": cheetah_a2c_progress}

########################### DDPG ###################################

cheetah_ddpg_progress = {"gamma": 0.99, "buffer_size": int(1e6), 
                         "tau": 0.01,
                         "hybrid_eta": 0.005,
                         "policy_kwargs": {"net_arch": [256, 256]},
                        "learning_rate_start": 1e-3,
                        "learning_rate_end": 1e-4,
                        "learning_fraction": 1., 
                        "batch_size": 100,
                        "adrs_update": 500,
                        "theta": 500,
                        "use_one_hot": True,
                        "eval_freq": 1000,
                        "total_timesteps": 2_000_000
                        }

cheetah_ddpg_naive = cheetah_ddpg_progress.copy()
cheetah_ddpg_naive.update({
    "theta": 0,
})

cheetah_ddpg_hybrid = cheetah_ddpg_progress.copy()

cheetah_ddpg_noise_progress = cheetah_ddpg_progress.copy()
cheetah_ddpg_noise_progress.update({
    "adrs_update": 5000,
    "theta": 100,
    "learning_rate_start": 1e-4,
    "learning_rate_end": 1e-5,
    "learning_fraction": 1.,
})

cheetah_ddpg_noise_naive = cheetah_ddpg_progress.copy()
cheetah_ddpg_noise_naive.update({
    "adrs_update": 1000,
    "theta": 0,
    "learning_rate_start": 1e-4,
    "learning_rate_end": 1e-4,
    "learning_fraction": 1.,
})
cheetah_ddpg_noise_hybrid = cheetah_ddpg_progress.copy()
cheetah_ddpg_noise_hybrid.update({
    "adrs_update": 2000,
    "theta": 500,
    "learning_rate_start": 1e-3,
    "learning_rate_end": 1e-5,
    "learning_fraction": 0.2,
})

cheetah_ddpg_normal = {"progress": cheetah_ddpg_progress, "hybrid": cheetah_ddpg_hybrid, "naive": cheetah_ddpg_naive}
cheetah_ddpg_noise = {"progress": cheetah_ddpg_noise_progress, "hybrid": cheetah_ddpg_noise_hybrid, "naive": cheetah_ddpg_noise_naive}
cheetah_ddpg_missing = cheetah_ddpg_normal.copy()

cheetah_ddpg_params = {}
cheetah_ddpg_params["noise"] = cheetah_ddpg_noise
cheetah_ddpg_params["normal"] = cheetah_ddpg_normal
cheetah_ddpg_params["missing"] = cheetah_ddpg_missing

####################################################################
# FUNCTION TO GET HYPERPARAMS
####################################################################

def get_param(env_name: str, reward_type: str, alg: str = "ddpg", env_type: str = "normal"):
    if env_name == "water":
        param = water_params[env_type]
        learning_params = ContiWorldLearningParams(**param[reward_type])

    elif env_name == "cheetah" and alg in ["ddpg", "td3", "sac"]:
        param = cheetah_ddpg_params[env_type]
        learning_params = ContiWorldLearningParams(**param[reward_type])
    
    elif env_name == "cheetah" and alg == "ppo":
        learning_params = ContiWorldLearningParams(**cheetah_ppo[reward_type])
    
    elif env_name == "cheetah" and alg == "a2c":
        learning_params = ContiWorldLearningParams(**cheetah_a2c[reward_type])

    elif env_name == "taxi":
        param = taxi_params[env_type]
        learning_params = GridWorldLearningParams(**param[reward_type])

    elif env_name == "office":
        param = office_params[env_type]
        learning_params = GridWorldLearningParams(**param[reward_type])

    elif env_name == "toy":
        learning_params = GridWorldLearningParams(**toy[reward_type])
    
    else:
        raise NotImplementedError("Each algorithm is used for a specific environment. Check utils.param_info.py")
    
    return learning_params
