import os
import glob
import pandas as pd
import numpy as np
from results_plot.plotter import load_npz


def get_ltl_results(
    env_name: str, 
    missing: bool, 
    noise: float, 
    window_size: int = 20, 
    reward_types: list = ["naive", "progress", "hybrid"], 
    theta=None,
    adrs_update=None,
    alg_name=None
    ):
    # cheetah_adrs/ddpg/hybrid
    assert noise in [0, 0.1]
    if missing:
        env_name += "_infeasible"
    if noise == 0.1:
        env_name += "_noise"
    results = []
    for reward_type in reward_types:
        files = []
        
        for i in range(10):
            if theta is not None:
                path = os.getcwd() + "/" + env_name + "/" + alg_name + f"/{reward_type}_theta{theta}_update{adrs_update}/" + str(i) + "/*.npz"
                if not os.path.exists(path):
                    path = os.getcwd() + "/" + env_name + "/" + alg_name + f"/{reward_type}_theta{theta}.0_update{adrs_update}/" + str(i) + "/*.npz"
            else:
                path = os.getcwd() + "/" + env_name + "/" + alg_name + "/" + reward_type + "/" + str(i) + "/*.npz"
            print(path)
            files += glob.glob(path)

        temp_results = []
        for i in range(len(files)):
            temp_df = {}
            s, ps, r, ep_lengths = load_npz(files[i], window_size)
            
            temp_df["Success Rate"] = list(s)
            temp_df["Partial Achieve"] = list(ps)
            temp_df["Average Return"] = list(r)
            
            if "progress" in reward_type:
                rt = "Adaptive Progression"
            elif "hybrid" in reward_type:
                rt = "Adaptive Hybrid"
            elif "naive" in reward_type:
                rt = "Naive"
            
            temp_df[""] = [rt] * len(s)
            temp_df["Training Steps"] = range(len(s))
            temp_results.append(temp_df)

        results += temp_results
    
    return results


def save_file(
    results, 
    plot_type, 
    env_name, 
    reward_type, 
    adrs_update,
    theta,
    alg_name=None
    ):
    # normalize factor for each environment
    env2maxr = {"office": 1.19, "taxi": 1.3, "water": 0.27, "cheetah": 5.}
    v = []
    for r in results:
        v.append(r["Success Rate"]) if plot_type == "sr" else v.append(r["Average Return"])

    if plot_type == "reward":
        v = np.array(v) / env2maxr[env_name]

    mean = np.mean(v, axis=0)
    std = np.std(v, axis=0)
    df = pd.DataFrame({"mean": mean, "std": std})
    folder_path = os.getcwd() + f"/{env_type}_csv"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    save_path = folder_path + f"/{plot_type}_{env_name}_{alg_name}_{reward_type}_theta{int(theta)}_update{adrs_update}.csv"
    df.to_csv(save_path)

        
type2name = {"sr": "Success Rate", "reward": "Average Return"}
alg_names = ["dqn", "ddqn"]
alg2env_name = {"dqn": "taxi", "ddqn": "water"}

missing = True
noise = False
env_type = "missing"
thetas = [2000, 5000, 10000]
adrs_updates = [50, 500, 5000]

for alg_name in alg_names:
    for theta in thetas:
        for adrs_update in adrs_updates:
            for plot_type in ["sr", "reward"]:
                env_name = alg2env_name[alg_name]
                try:
                    reward_types = ["progress_adrs", "hybrid_adrs"]
                    for r in reward_types:
                        results = get_ltl_results(env_name, missing, noise, reward_types=[r], alg_name=alg_name, adrs_update=adrs_update, theta=theta)
                        save_file(results, plot_type, env_name, reward_type=r, alg_name=alg_name, adrs_update=adrs_update, theta=theta)

                except Exception as e:
                    print(alg_name + " does not work")
                    print(e.args)
                    pass 
                