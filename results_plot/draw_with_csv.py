import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter
import sys

env_type = sys.argv[1]
assert env_type in ['normal', 'missing', 'noise'], 'env type should be one of normal, missing, noise'

try:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
except:
    print("latex could not be found")

formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 2))  # Adjust power limits as needed

alg2color = {
            "crm": (0.7686274509803922, 0.3058823529411765, 0.3215686274509804), \
            "qrm": (0.8666666666666667, 0.5176470588235295, 0.3215686274509804), \
            "hrm": (0.8, 0, 0.8), 
            "naive": (0.5058823529411764, 0.4470588235294118, 0.7019607843137254), \
            "progress_adrs": (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),\
            "hybrid_adrs":(0.2980392156862745, 0.4470588235294118, 0.6901960784313725),\
            "naive_adrs": (0.5058823529411764, 0.4470588235294118, 0.7019607843137254), \
            }

env2step = {"office": 100, "taxi": 1000, "water": 1000, "cheetah": 1000}
env2total_step = {"office": 6e2,  "taxi": 5e2, "water": 2e3, "cheetah": 2e3}
env2title = {"office": "Office World", "taxi": "Taxi World", "water": "Water World", "cheetah": "HalfCheetah"}
env2algo = {"office": "dqn", "taxi": "dqn", "water": "ddqn", "cheetah": "ddpg"}

if env_type == "missing":
    row2type = {0: "reward"}
    row2yaxis = {0: "Normalized Reward"}
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 2))
    row_nbr = 1
else:
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 4), gridspec_kw={'hspace': 0.5})
    row2type = {0: "sr", 1: "reward"}
    row2yaxis = {0: "Success Rate", 1: "Normalized Reward"}
    row_nbr = 2


def draw_graph(env_name, ax, plot_type, row, col, noise=False, missing=False):
    alg_names = ["qrm", "crm", "hrm", "naive", "progress_adrs", "hybrid_adrs"]
    for alg_name in alg_names:
        try:
            if alg_name not in ["qrm", "hrm", "crm"]:
                if noise:
                    df = pd.read_csv(f"./{env_type}_csv/{plot_type}_{env_name}_noise_{env2algo[env_name]}_{alg_name}.csv")
                elif missing:
                    df = pd.read_csv(f"./{env_type}_csv/{plot_type}_{env_name}_missing_{env2algo[env_name]}_{alg_name}.csv")
                else:
                    df = pd.read_csv(f"./{env_type}_csv/{plot_type}_{env_name}_normal_{env2algo[env_name]}_{alg_name}.csv")
            else:
                if noise:
                    df = pd.read_csv(f"./{env_type}_csv/{plot_type}_{env_name}_noise_{alg_name}_rm.csv")
                elif missing:
                    df = pd.read_csv(f"./{env_type}_csv/{plot_type}_{env_name}_missing_{alg_name}_rm.csv")
                else:
                    df = pd.read_csv(f"./{env_type}_csv/{plot_type}_{env_name}_normal_{alg_name}_rm.csv")
            # Sample data
            mean = df["mean"]
            std = df["std"]
            
            x = np.arange(len(mean)) * env2step[env_name] # X-axis values
            y = np.array(mean)   # Mean or median values
            std_dev = np.array(std)  # Standard deviation values
            
            # Plot the mean or median line
            l = alg_name.upper()
            
            if alg_name == "naive":
                l = "Naive"
            if alg_name == "naive_adrs":
                l = "Adaptive Naive"
            if alg_name == "progress_adrs":
                l = "Adaptive Progression"
            if alg_name == "hybrid_adrs":
                l = "Adaptive Hybrid"
            if alg_name == "qrm":
                l = "QRM"
            if alg_name == "crm":
                l = "CRM"
            if alg_name == "hrm":
                l = "HRM"
            
            c = alg2color[alg_name]
            ax.plot(x, y, label=l, linestyle='-', color=c)

            # Plot error bars using standard deviation
            ax.fill_between(x, y - std_dev, y + std_dev, alpha=0.2, color=c) # , label='Std Dev'
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
            # Add labels and legend
            ax.set_xlabel('Training Steps', fontdict={"fontsize": 12, "fontname" :"Helvetica"})
            if col == 0:
                ax.set_ylabel(row2yaxis[row], fontdict={"fontsize": 14, "fontname" :"Helvetica"})

            if env_name in ["cheetah", "water"]:
                ax.set_xticks([0, 1000000, 2000000])
            ax.set_yticks([0, 0.5, 1])
            ax.set_ylim(0, 1)
            ax.set_xlim(min(x), env2total_step[env_name] * env2step[env_name])
            ax.xaxis.set_major_formatter(formatter)

            if row == 0:
                ax.set_title(env2title[env_name], fontsize=18, fontname="Helvetica")
            if row == 0 and col == 0:
                if env_type == "missing":
                    ax.legend(loc='upper center', bbox_to_anchor=(2.3, 1.6), ncol=6, prop = { "size": 14 })
                else:
                    ax.legend(loc='upper center', bbox_to_anchor=(2.3, 1.8), ncol=6, prop = { "size": 14 })
                pass
        except Exception as error:
            print(error)

for row in range(row_nbr):
    for col, env_name in enumerate(["office", "taxi", "water", "cheetah"]):
        if env_type == "normal":
            draw_graph(env_name, axes[row, col], row2type[row], row, col, noise = False)
        elif env_type == "noise":
            draw_graph(env_name, axes[row, col], row2type[row], row, col, noise = True)    
        else:
            draw_graph(env_name, axes[col], row2type[row], row, col, missing=True)

plt.grid(False)
plt.savefig(f"./saved_plots/{env_type}.png", dpi=600, bbox_inches="tight")
plt.show()

