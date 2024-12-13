import numpy as np
import os, argparse
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import sys
from stable_baselines3.common.results_plotter import rolling_window


def load_json(path, num_times):
    with open(path, "r") as f:
        results = json.load(f)

    f = lambda x: np.array(x).reshape(num_times, -1)
    eval_rewards = f(results["results"]["eval_rewards"])
    mdp_rewards = f(results["results"]["mdp_rewards"])
    rm_states = f(results["results"]["rm_states"])
    last_rm_states = f(results["results"]["last_rm_states"])
    ep_lengths = None
    # ep_lengths = f(results["results"]["ep_lengths"])

    print(results["world"])
    return eval_rewards, mdp_rewards, rm_states, last_rm_states, ep_lengths

def draw(results: list, type: str = "sr"):
    # results must be [[run 1 result (dict)], [run 2 result (dict)], [run 3 result (dict)]]
    if type == "sr":
        df = {"Success Rate": [], "Training Steps": [], "Algorithms": []}
    elif type == "ps":
        df = {"Partial Achieve": [], "Training Steps": [], "Algorithms": []}
    else:
        df = {"Reward": [], "Training Steps": [], "Algorithms": []}

    for result in results:
        # print(result)
        df["Training Steps"] += result["Training Steps"]
        if type == "sr":
            df["Success Rate"] += result["Success Rate"]
        elif type == "ps":
            df["Partial Achieve"] += result["Partial Achieve"]
        else:
            df["Reward"] += result["Reward"]

        df["Algorithms"] += result["Algorithms"]
    
    if type == "sr":
        sns.relplot(
            data=df, kind="line",
            x="Training Steps", y="Success Rate", hue="Algorithms"
        )
    elif type == "ps":
        sns.relplot(
            data=df, kind="line",
            x="Training Steps", y="Partial Achieve", hue="Algorithms"
        )
    else:
        sns.relplot(
            data=df, kind="line",
            x="Training Steps", y="Reward", hue="Algorithms"
        )
    plt.show()


def list2dict(multi_run_results, plot_type: str = "sr", alg_name: str = "qrm"):
    results = []
    for each_run_result in multi_run_results:
        temp_df = {}
        each_run_result = rolling_window(each_run_result, window=20)
        each_run_result = np.mean(each_run_result, axis=1)
        if plot_type == "sr":
            temp_df["Success Rate"] = list(each_run_result)
        elif plot_type == "ps":
            temp_df["Partial Achieve"] = list(each_run_result)
        if plot_type == "reward":
            temp_df["Reward"] = list(each_run_result)
        else:
            ValueError("plot type is not proper")
        temp_df["Algorithms"] = [alg_name] * len(each_run_result)
        temp_df["Training Steps"] = range(len(each_run_result))
        results.append(temp_df)
    return results

if __name__ == "__main__":
    env_name = sys.argv[1]
    alg_name = sys.argv[2] # qrm, qrm-rs
    map_id = sys.argv[3]
    plot_type = sys.argv[4]
    path = "./" + env_name + "/" + env_name + "/" + alg_name + ".json"
    assert env_name in ["water", "taxi", "office"]
    assert plot_type in ["reward", "sr", "ps"]

    if env_name == "water":
        path = "./" + env_name + "/" + env_name + "_" + map_id + "/" + alg_name + ".json"
    
    eval_rewards, mdp_rewards, rm_states, last_rm_states, epi_lengths = load_json(path, 8)
    if plot_type == "sr":
        results = list2dict(rm_states, plot_type, alg_name)
    elif plot_type == "reward":
        results = list2dict(eval_rewards, plot_type, alg_name)
    else:
        raise ValueError

    
    draw(results, plot_type)
