from psltl.baseline_algo.qrm.src.tester.tester import Tester
from psltl.baseline_algo.qrm.src.tester.test_utils import get_precentiles
import numpy as np
import os, argparse
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from stable_baselines3.common.results_plotter import rolling_window

def export_results_water_world():
    # NOTE: We do not report performance on Map 0 because we used it as our validation map
    maps = ["water_%d"%i for i in range(1,11)]

    world = "water"    
    tmp_folder = "../tmp/"
    algs = ["dqn", "hrl", "hrl-rm", "qrm", "qrm-rs"]

    results = {}
    for alg in algs:
        results[alg] = []
    
    for map_name in maps:
        # computing best known performance
        optimal = {}
        for alg in algs:
            result_file = os.path.join(tmp_folder, world, map_name, alg + ".json")
            tester = Tester(None, None, None, None, result_file)

            f_optimal = tester.get_best_performance_per_task()
            for t in f_optimal:
                if t not in optimal:
                    optimal[t] = f_optimal[t]
                else:
                    optimal[t] = max([optimal[t], f_optimal[t]])
        
        # adding results for this map to the result summary
        for alg in algs:
            result_file = os.path.join(tmp_folder, world, map_name, alg + ".json")
            tester = Tester(None, None, None, None, result_file)
            tester.world.optimal = optimal
            alg_results = tester.get_result_summary()["all"]

            for i in range(len(alg_results)):
                step, reward = alg_results[i]
                if len(results[alg]) == i:
                    results[alg].append((step, []))
                results[alg][i][1].append(reward)

    # Compute final stats and export summary file
    for alg in algs:
        folder_out = os.path.join("../tmp/results", world)
        if not os.path.exists(folder_out): os.makedirs(folder_out)
        f_out = open(os.path.join(folder_out, alg + ".txt"), "w")
        for i in range(len(results[alg])):
            step, reward = results[alg][i]
            p25, p50, p75 = get_precentiles(np.concatenate(reward))
            f_out.write(str(step) + "\t" + str(p25) + "\t" + str(p50) + "\t" + str(p75) + "\n")
        f_out.close()


def export_results_tabular_world(world, maps):
    tmp_folder = "../tmp/"
    algs = ["dqn", "hrl", "hrl-rm", "qrm", "qrm-rs"]

    results = {}
    for alg in algs:
        results[alg] = []
        
    for map_name in maps:
        # adding results for this map to the result summary
        for alg in algs:
            result_file = os.path.join(tmp_folder, world, map_name, alg + ".json")            
            tester = Tester(None, None, None, None, result_file)

            alg_results = tester.get_result_summary()["all"]

            for i in range(len(alg_results)):
                step, reward = alg_results[i]
                if len(results[alg]) == i:
                    results[alg].append((step, []))
                results[alg][i][1].append(reward)

    # Compute final stats and export summary file
    for alg in algs:
        folder_out = os.path.join("../tmp/results", world)
        if not os.path.exists(folder_out): os.makedirs(folder_out)
        f_out = open(os.path.join(folder_out, alg + ".txt"), "w")
        for i in range(len(results[alg])):
            step, reward = results[alg][i]
            p25, p50, p75 = get_precentiles(np.concatenate(reward))
            f_out.write(str(step) + "\t" + str(p25) + "\t" + str(p50) + "\t" + str(p75) + "\n")
        f_out.close()

def export_results_office_world():
    export_results_tabular_world("office", ["office"])

def export_results_craft_world():
    # NOTE: We do not report performance on Map 0 because we used it as our validation map
    export_results_tabular_world("craft", ["craft_%d"%i for i in range(1,11)])

def plot_saved_results(env_name: str, alg_name: str, map_id: int = None, num_times: int =1):
    if map_id != None:
        map = env_name + "_%d"%map_id
    else:
        map = env_name

    world = env_name
    tmp_folder = "../tmp/"

    results = {}
    
    result_file = os.path.join(tmp_folder, world, map, alg_name + ".json")
    with open(result_file, "r") as f:
        results = json.load(f)

    f = lambda x: np.array(x).reshape(num_times, -1)
    eval_rewards = f(results["results"]["eval_rewards"])
    mdp_rewards = f(results["results"]["mdp_rewards"])
    rm_states = f(results["results"]["rm_states"])
    last_rm_states = f(results["results"]["last_rm_states"])

    print("eval rewards", eval_rewards)
    print()
    print("mdp rewards", mdp_rewards)
    print()
    print("rm_states", rm_states)
    print()
    print("last rm states", last_rm_states)
    print()
    print(results["world"])
    return eval_rewards, mdp_rewards, rm_states, last_rm_states


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
    # EXAMPLE: python3 export_summary.py --world="craft"
    # python export_summary.py --world water --alg_name qrm --map_id 3
    # python export_summary.py --world taxi --alg_name qrm

    # Getting params
    worlds     = ["office", "craft", "water", "taxi"]

    parser = argparse.ArgumentParser(prog="export_summary", description='After running the experiments, this algorithm computes a summary of the results.')
    parser.add_argument('--world', default='office', type=str, 
                        help='This parameter indicated which world to solve. The options are: ' + str(worlds))
    parser.add_argument('--alg_name', default='qrm', type=str, 
                        help='This parameter indicated which world to solve. The options are: ' + str(worlds))
    parser.add_argument('--map_id', default=None, type=int, 
                        help='This parameter indicated which world to solve. The options are: ' + str(worlds))
    parser.add_argument('--num_times', default=1, type=int, 
                        help='This parameter indicated which world to solve. The options are: ' + str(worlds))

    args = parser.parse_args()
    if args.world not in worlds: raise NotImplementedError("World " + str(args.world) + " hasn't been defined yet")

    # Computing the experiment summary
    world = args.world
    alg_name = args.alg_name
    map_id = args.map_id
    num_times = int(args.num_times)
    if args.map_id is not None:
        map_id = int(map_id)


    eval_rewards, mdp_rewards, rm_states, last_rm_states = plot_saved_results(world, alg_name, map_id, num_times)
    plot_type = "reward" # ps, sr, reward
    eval_rewards = np.array(eval_rewards).reshape(8, -1)
    srs = np.array(rm_states).reshape(8, -1)
    results = list2dict(srs, plot_type, alg_name)
    
    draw(results, plot_type)

    # if world == "office":
    #     export_results_office_world()
    # if world == "craft":
    #     export_results_craft_world()
    # if world == "water":
    #     export_results_water_world()
