import os
import sys
import pandas as pd
import numpy as np
from results_plot.plotter import get_crm_results, get_ltl_results, get_qrm_results, save_file

# To get results on normal
# python compute_mean_std.py False 0

# To get results on missing
# python compute_mean_std.py True 0

# To get results on noise
# python compute_mean_std.py False 0.1

missing = eval(sys.argv[1])
noise = float(sys.argv[2])

if not missing and noise <= 0:
    env_type = "normal"
elif missing:
    env_type = "missing"
elif noise > 0:
    env_type = "noise"

type2name = {"sr": "Success Rate", "reward": "Average Return"}
alg_names = ["crm", "hrm", "qrm", "ppo", "a2c", "ddpg", "dqn", "ddqn"]
# alg_names = ["hrm"]
envs = ["office", "taxi", "water", "cheetah"]

for alg_name in alg_names:
    for plot_type in ["sr", "reward"]:
        for env_name in envs:
            try:
                # env_name: str, missing: bool, noise: float,
                if alg_name == "crm":
                    results = get_crm_results(env_name, missing, noise, algs=["crm-rs"])
                    save_file(results, plot_type, env_name, env_type, "rm", alg_name="crm")
                # hrm result shared by get crm results function
                elif alg_name == "hrm":
                    results = get_crm_results(env_name, missing, noise, algs=["hrm-rs"])
                    save_file(results, plot_type, env_name, env_type, "rm", alg_name="hrm")
                # get qrm results come from different format
                elif alg_name == "qrm":
                    results = get_qrm_results(env_name, missing, noise, plot_type=plot_type, algs=["qrm-rs"])
                    save_file(results, plot_type, env_name, env_type, "rm", alg_name="qrm")
                else:
                    reward_types = ["progress_adrs", "hybrid_adrs", "naive"]
                    for r in reward_types:
                        results = get_ltl_results(env_name, missing, noise, reward_types=[r], alg_name=alg_name)
                        save_file(results, plot_type, env_name, reward_type=r, env_type=env_type, alg_name=alg_name)

            except Exception as e:
                print(alg_name + " does not work")
                print(e.args)
                pass 
            