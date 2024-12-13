import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import pickle5 as pickle
import sys
from stable_baselines3.common.results_plotter import rolling_window

sns.set(style="white", font_scale=2)

def draw(results: list, type: str = "sr", title: str = "office"):
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
    plt.title(title)
    plt.show()


def load_npz(path, window_size=20):
    with np.load(path) as data:
        successes = data["successes"]
        partial_successes = data["partial_successes"]
        rewards = data["results"]
        rewards = rewards[:, -1].reshape(-1, 1)
        
        s = np.mean(successes, axis=1)
        ps = np.mean(partial_successes, axis=1)
        r = np.mean(rewards, axis=1)
        ep_lenghts = np.mean(data["ep_lengths"], axis=1)
        if len(s) > 10000:
            new_s = [s[i] for i in range(0, 20000, 10)]
            new_r = [r[i] for i in range(0, 20000, 10)]

            s = np.array(new_s)
            r = np.array(new_r)

        org_s = s
        org_ps = ps
        org_r = r

        s = rolling_window(s, window=window_size)
        s = np.mean(s, axis=1)
        
        ps = rolling_window(ps, window=window_size)
        ps = np.mean(ps, axis=1)

        r = rolling_window(r, window=window_size)
        r = np.mean(r, axis=1)

        pre_s = []
        for i in range(1, window_size):
            pre_s = np.append(pre_s, np.mean(org_s[:i]))
        s = np.append(pre_s, s)

        pre_ps = []
        for i in range(1, window_size):
            pre_ps = np.append(pre_ps, np.mean(org_ps[:i]))
        ps = np.append(pre_ps, ps)

        pre_r = []
        for i in range(1, window_size):
            pre_r = np.append(pre_r, np.mean(org_r[:i]))
        r = np.append(pre_r, r)

    return s, ps, r, ep_lenghts


########################### BELOWS ARE FUNCTION FOR QRM #####################################

def return_results(file_paths, algs, window_size=20):
    results = []
    for idx, file_path in enumerate(file_paths):
        for i in range(len(file_path)):
            temp_df = {}
            s, ps, r, ep_lengths = load_npz(file_path[i], window_size)

            temp_df["Success Rate"] = list(s)
            temp_df["Partial Achieve"] = list(ps)
            temp_df["Reward"] = list(r)
            temp_df["Algorithms"] = [algs[idx]] * len(s)
            temp_df["Training Steps"] = range(len(s))
            results.append(temp_df)
    return results



if __name__ == "__main__":
    env_name = sys.argv[1]
    alg = "crm"
    plot_type = sys.argv[2]
    missing = eval(sys.argv[3])
    assert plot_type in ["sr", "ps", "reward"]
    if missing:
        # file_path = glob.glob("./" + env_name + "_results" + "/" + alg + "_missing/*")
        # file_path_2 = glob.glob("./" + env_name + "_results" + "/" + alg + "_rs" + "_missing/*")
        file_path = glob.glob("./" + env_name + "/" + alg + "_missing/*")
        file_path_2 = glob.glob("./" + env_name + "/" + alg + "_rs" + "_missing/*")
    else:
        file_path = glob.glob("./" + env_name + "_results" + "/" + alg + "/*")
        file_path_2 = glob.glob("./" + env_name + "_results" + "/" + alg + "_rs" + "/*")
    file_paths = [file_path, file_path_2]
    algs = ["crm", "crm-rs"]
    results = return_results(file_paths, algs, window_size=20)
    draw(results, plot_type)
