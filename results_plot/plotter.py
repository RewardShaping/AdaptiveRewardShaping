import os
import seaborn as sns
import numpy as np
import glob
import json
from stable_baselines3.common.results_plotter import rolling_window
import pandas as pd

sns.set_theme(style="white", font_scale=2)

def draw_common(results:list, Type: str= "sr", env_name: str = "office", rl_algo_name: str = "dqn"):
    df = {"Success Rate": [], "Training Steps": [], "": [], "Type": []}
    for result in results:
        # print(result)
        df["Training Steps"] += result["Training Steps"]
        if Type == "sr":
            df["Success Rate"] += result["Success Rate"]
        else:
            df["Success Rate"] += result["Average Return"]

        df[""] += result[""]
    
    df["Env"] = [env_name] * len(df["Training Steps"])
    df["RLAlgo"] = [rl_algo_name] * len(df["Training Steps"])
    df["Type"] = [Type] * len(df["Training Steps"])

    return df

def draw(results: list, type: str = "sr", env_name: str = "office", rl_algo_name: str = "dqn", style: str = "ltl"):
    # results must be [[run 1 result (dict)], [run 2 result (dict)], [run 3 result (dict)]]
    if type == "sr":
        df = {"Success Rate": [], "Training Steps": [], "": []}
    elif type == "ps":
        df = {"Partial Achieve": [], "Training Steps": [], "": []}
    else:
        df = {"Average Return": [], "Training Steps": [], "": []}
    # df = defaultdict(list)
    df["FSR"] = []
    df["FAR"] = []
    for result in results:
        # print(result)
        df["Training Steps"] += result["Training Steps"]
        if type == "sr":
            df["Success Rate"] += result["Success Rate"]
        elif type == "ps":
            df["Partial Achieve"] += result["Partial Achieve"]
        else:
            df["Average Return"] += result["Average Return"]

        df[""] += result[""]
        try:
            df["FSR"] += [result["FSR"]]
            df["FAR"] += [result["FAR"]]
        except:
            pass
    
    df["Env"] = [env_name] * len(df["Training Steps"])
    df["RLAlgo"] = [rl_algo_name] * len(df["Training Steps"])
    df["STYLE"] = [style] * len(df["Training Steps"])
    return df
    
def load_npz(path, window_size):
    with np.load(path, allow_pickle=True) as data:
        successes = data["successes"]
        partial_successes = data["partial_successes"]
        rewards = data["results"]
        successes = successes[1:]
        partial_successes = partial_successes[1:]
        # for crm case, we have already average the results..
        if len(np.shape(rewards)) == 1:
            if (type(rewards[1]) != list):
                rewards = rewards.reshape(-1, 1)
            # for cheetah crm
            else:
                rewards = rewards[1:]
                new_rewards = [sum(x) / 5 for x in rewards]
                rewards = np.array(new_rewards)
                rewards = rewards.reshape(-1, 1)
                
                new_sr = [sum(x) / 5 for x in successes]
                new_sr = np.array(new_sr)
                successes = new_sr.reshape(-1, 1)

                new_ps = [sum(x) / 5 for x in partial_successes]
                new_ps = np.array(new_ps)
                partial_successes = new_ps.reshape(-1, 1)
        else:
            rewards = rewards[:, -1].reshape(-1, 1)
        
        # print(rewards)
        # print(np.shape(rewards))
        
        s = np.mean(successes, axis=1)
        ps = np.mean(partial_successes, axis=1)
        r = np.mean(rewards, axis=1)

        if len(s) < 50:
            return [], [], [], []
        
        if len(s) > 4000:
            new_s = [s[i] for i in range(0, len(s), 10)]
            new_r = [r[i] for i in range(0, len(s), 10)]

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

    ep_lengths = []

    return s, ps, r, ep_lengths


############################# BELOWS ARE FUNCTIONS FOR CRM ##################################
def return_results(file_paths, algs, window_size=20):
    results = []
    for idx, file_path in enumerate(file_paths):
        for i in range(len(file_path)):
            temp_df = {}
            s, ps, r, ep_lengths = load_npz(file_path[i], window_size)
            
            temp_df["Success Rate"] = list(s)
            temp_df["Partial Achieve"] = list(ps)
            temp_df["Average Return"] = list(r)
            if "crm" in algs[idx]:
                algo_name = "CRM"
            if "hrm" in algs[idx]:
                algo_name = "HRM"
            # temp_df[""] = [algs[idx]] * len(s) # if you want to distinguish crm and crm-rs, use this
            temp_df[""] = [algo_name] * len(s)
            temp_df["Training Steps"] = range(len(s))
            temp_df["FSR"] = [list(s)[-1]] # the last success rate we measured
            temp_df["FAR"] = [list(r)[-1]] # the last reward we measured
            results.append(temp_df)
    
    return results


############################# BELOWS ARE FUNCTIONS FOR QRM ##################################
def load_json(path, num_times):
    with open(path, "r") as f:
        results = json.load(f)

    f = lambda x: np.array(x).reshape(num_times, -1)
    eval_rewards = f(results["results"]["eval_rewards"])
    mdp_rewards = f(results["results"]["mdp_rewards"])
    rm_states = f(results["results"]["rm_states"])
    last_rm_states = f(results["results"]["last_rm_states"])
    ep_lengths = f(results["results"]["ep_lengths"])

    return eval_rewards, mdp_rewards, rm_states, last_rm_states, ep_lengths

def rmStates2Success(rm_states, env_name):
    if env_name == "taxi":
        f = lambda x: 1 if x == 3 else 0
    elif env_name == "office":
        f = lambda x: 1 if x == 1 else 0
    elif env_name == "water":
        f = lambda x: 1 if x == 0 else 0

    pre_successes = []
    for rm_state in rm_states:
        y = map(f, rm_state)
        pre_successes.append(list(y))
    pre_successes = np.array(pre_successes)

    return pre_successes

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
            temp_df["Average Return"] = list(each_run_result)
        elif plot_type == "violation":
            temp_df["Violation"] = list(each_run_result)
        else:
            ValueError("plot type is not proper")
        
        if "qrm" in alg_name:
            alg_name = "QRM"
        temp_df[""] = [alg_name] * len(each_run_result)
        temp_df["Training Steps"] = range(len(each_run_result))
        results.append(temp_df)

    return results


#################################### This function is to plot all results together ###############################
def get_crm_results(env_name: str, missing: bool, noise: float, window_size: int = 20, algs=["crm", "crm-rs"]):
    ############################################ 
    # crm and hrm, both are using rs
    ############################################
    assert noise in [0, 0.1, 0.2]
    folder_name = "normal"
    if missing:
        folder_name = "missing"
    if noise == 0.1:
        folder_name = "noise01"

    results = []
    file_paths = []
    try:
        if "crm-rs" in algs:
            crm_rs = glob.glob("./" + folder_name + "/" + env_name + "/crm_rs/*.npz")
            file_paths.append(crm_rs)
        if "hrm-rs" in algs:
            hrm_rs = glob.glob("./" + folder_name + "/" + env_name + "/hrm_rs/*.npz")
            file_paths.append(hrm_rs)
    except:
        print("env name:", env_name)
        print("folder name:", folder_name)
        print("crm found no results")
        pass

    results += return_results(file_paths, algs, window_size)

    return results

def get_qrm_results(env_name: str, missing: bool, noise: float, window_size: int = 20, algs=["qrm", "qrm-rs"], plot_type = "sr"):
    ############################################ 
    # plot results of qrm 
    ############################################
    assert noise in [0, 0.1, 0.2]
    folder_name = "normal"
    if missing:
        folder_name = "missing"
    if noise == 0.1:
        folder_name = "noise01"

    results = []
    files_paths = []
    try:
        if "qrm-rs" in algs:
            qrm = "./" + folder_name + "/" + env_name + "/qrm_rs/qrm_rs.json"
            files_paths.append(qrm)
    except:
        print("qrm found no results")
        pass

    if not env_name == "cheetah":
        for alg_name, path in zip(algs, files_paths):
            naive_rewards, mdp_rewards, rm_states, last_rm_states, epi_lengths = load_json(path, 10) # 10 is number of total run
            pre_successes = rmStates2Success(rm_states, env_name)
            if plot_type == "sr":
                results += list2dict(pre_successes, plot_type, alg_name, window_size)
                
            elif plot_type == "reward":
                results += list2dict(naive_rewards, plot_type, alg_name, window_size)
            else:
                raise ValueError
    else:
        pass
            
    return results

def get_ltl_results(
        env_name: str, 
        missing: bool, 
        noise: float, 
        window_size: int = 20, 
        reward_types: list = ["naive", "progress", "hybrid"], 
        alg_name=None
        ):
    # cheetah_adrs/ddpg/hybrid
    assert noise in [0, 0.1]
    folder_name = "normal"
    if missing:
        folder_name = "missing"
    if noise == 0.1:
        folder_name = "noise01"
        
    results = []
    
    for reward_type in reward_types:
        files = []
        for i in range(10):
            if alg_name in ["a2c", "ppo"]:
                path = "./" + folder_name + "/" + env_name + "/" + alg_name + "/" + reward_type + "/" + str(i) + "/*.npz"
            else:
                path = "./" + folder_name + "/" + env_name + "/" + reward_type + "/" + str(i) + "/*.npz"
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
    env_type,
    reward_type, 
    alg_name,
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
    env_name += "_" + env_type
    
    df = pd.DataFrame({"mean": mean, "std": std})
    folder_path = os.getcwd() + f"/{env_type}_csv"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    df.to_csv(folder_path + f"/{plot_type}_{env_name}_{alg_name}_{reward_type}.csv")
