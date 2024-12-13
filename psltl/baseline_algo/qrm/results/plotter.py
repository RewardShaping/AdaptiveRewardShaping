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
    ep_lengths = f(results["results"]["ep_lengths"])

    print(results["world"])
    return eval_rewards, mdp_rewards, rm_states, last_rm_states, ep_lengths

def draw(results: list, type: str = "sr"):
    # results must be [[run 1 result (dict)], [run 2 result (dict)], [run 3 result (dict)]]
    if type == "sr":
        df = {"Success Rate": [], "Training Steps": [], "Algorithms": []}
    elif type == "ps":
        df = {"Partial Achieve": [], "Training Steps": [], "Algorithms": []}
    elif type == "reward":
        df = {"Reward": [], "Training Steps": [], "Algorithms": []}
    else:
        df = {"Violation": [], "Training Steps": [], "Algorithms": []}

    for result in results:
        # print(result)
        df["Training Steps"] += result["Training Steps"]
        if type == "sr":
            df["Success Rate"] += result["Success Rate"]
        elif type == "ps":
            df["Partial Achieve"] += result["Partial Achieve"]
        elif type == "reward":
            df["Reward"] += result["Reward"]
        else:
            df["Violation"] += result["Violation"]

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
    elif type == "reward":
        sns.relplot(
            data=df, kind="line",
            x="Training Steps", y="Reward", hue="Algorithms"
        )
    elif type == "violation":
        sns.relplot(
            data=df, kind="line",
            x="Training Steps", y="Violation", hue="Algorithms"
        )

    plt.show()


def list2dict(multi_run_results, plot_type: str = "sr", alg_name: str = "qrm", window_size=20):
    results = []
    for each_run_result in multi_run_results:
        temp_df = {}
        org_each_run_result = each_run_result
        each_run_result = rolling_window(each_run_result, window=window_size)
        each_run_result = np.mean(each_run_result, axis=1)
        pre_result = []

        for i in range(1, window_size):
            pre_result = np.append(pre_result, np.mean(org_each_run_result[: i]))
        
        each_run_result = np.append(pre_result, each_run_result)
        
        if plot_type == "sr":
            temp_df["Success Rate"] = list(each_run_result)
        elif plot_type == "ps":
            temp_df["Partial Achieve"] = list(each_run_result)
        elif plot_type == "reward":
            temp_df["Reward"] = list(each_run_result)
        elif plot_type == "violation":
            temp_df["Violation"] = list(each_run_result)
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
    is_missing = eval(sys.argv[5])
    if is_missing:
        path = "./missing/" + env_name + "/" + env_name + "/" + alg_name + ".json"
    else:
        path = "./complete/" + env_name + "/" + env_name + "/" + alg_name + ".json"
    assert env_name in ["water", "taxi", "office"]
    assert plot_type in ["reward", "sr", "ps", "violation"]

    if env_name == "water":
        if is_missing:
            path = "./missing/" + env_name + "/" + env_name + "_" + map_id + "/" + alg_name + ".json"
        else:
            path = "./complete/" + env_name + "/" + env_name + "_" + map_id + "/" + alg_name + ".json"
    
    eval_rewards, mdp_rewards, rm_states, last_rm_states, epi_lengths = load_json(path, 10)
    if env_name == "taxi":
        f = lambda x: 1 if x == 3 else 0
        g = lambda x: 1 if x == 4 else 0
    elif env_name == "office":
        f = lambda x: 1 if x == 1 else 0
        g = lambda x: 1 if x == 4 else 0
    elif env_name == "water":
        f = lambda x: 1 if x == 0 else 0
        g = lambda x: 1 if x == 4 else 0

    pre_successes = []
    for rm_state in rm_states:
        y = map(f, rm_state)
        pre_successes.append(list(y))
    pre_violation = []
    for last_rm_state in last_rm_states:
        y = map(g, last_rm_state)
        pre_violation.append(list(y))

    pre_successes = np.array(pre_successes)
    pre_violation = np.array(pre_violation)

    if plot_type == "sr":
        results = list2dict(pre_successes, plot_type, alg_name)
    elif plot_type == "reward":
        results = list2dict(eval_rewards, plot_type, alg_name)
    elif plot_type == "ps":
        results = list2dict(rm_states, plot_type, alg_name)
    elif plot_type == "violation":
        results = list2dict(pre_violation, plot_type, alg_name)
    
    draw(results, plot_type)
