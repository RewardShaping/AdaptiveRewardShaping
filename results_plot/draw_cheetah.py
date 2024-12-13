import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter

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

# Create a figure and axis
alg2color = {"crm": (0.7686274509803922, 0.3058823529411765, 0.3215686274509804), 
             "qrm": (0.8666666666666667, 0.5176470588235295, 0.3215686274509804), \
            "naive": (0.5058823529411764, 0.4470588235294118, 0.7019607843137254), \
            "progress_adrs": (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),\
            "hybrid_adrs":(0.2980392156862745, 0.4470588235294118, 0.6901960784313725),\
            "progress": (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),\
            "hybrid":(0.2980392156862745, 0.4470588235294118, 0.6901960784313725),\
            }

env2step = {"office": 100, "taxi": 1000, "water": 1000, "cheetah": 1000}
env2total_step = {"office": 6e2,  "taxi": 5e2, "water": 2e3, "cheetah": 2e3}
env2title = {"normal": "Deterministic".upper(), "noise": "Noisy".upper(), "missing": "Infeasible".upper()}

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(9, 4.5), gridspec_kw={'hspace': 1.1})
row2type = {0: "reward", 1: "reward", 2: "reward"}
row2yaxis = {0: "Normalized Reward", 1: "Normalized Reward", 2: "Normalized Reward"}
row_nbr = 3
row2rl_algo = {0: "ddpg", 1: "ppo", 2: "a2c"}

def draw_graph(env_type, ax, row, col):
    rl_alg_name = row2rl_algo[row]
    noise, missing = False, False
    if env_type == "noise":
        noise = True
    elif env_type == "missing":
        missing = True

    plot_type = "reward"
    env_name = "cheetah"

    for alg_name in ["naive", "progress_adrs", "hybrid_adrs"]:
        try:
            if noise:
                df = pd.read_csv(f"./{env_type}_csv/{plot_type}_{env_name}_noise_{rl_alg_name}_{alg_name}.csv")
            elif missing:
                df = pd.read_csv(f"./{env_type}_csv/{plot_type}_{env_name}_missing_{rl_alg_name}_{alg_name}.csv")
            else:
                df = pd.read_csv(f"./{env_type}_csv/{plot_type}_{env_name}_normal_{rl_alg_name}_{alg_name}.csv")
            
            # Sample data
            mean = df["mean"]
            std = df["std"]

            x = np.arange(len(mean)) * 1000 # X-axis values
            y = np.array(mean)   # Mean or median values
            std_dev = np.array(std)  # Standard deviation values
            
            # Plot the mean or median line
            l = alg_name.upper()
            if alg_name == "naive":
                l = "Naive"
            if alg_name == "progress_adrs":
                l = "Adaptive Progression"
            if alg_name == "hybrid_adrs":
                l = "Adaptive Hybrid"

            c = alg2color[alg_name]
            ax.plot(x, y, label=l, linestyle='-', color=c)

            # Plot error bars using standard deviation
            ax.fill_between(x, y - std_dev, y + std_dev, alpha=0.2, color=c) # , label='Std Dev'
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Add labels and legend
            ax.set_xlabel('Training Steps', fontdict={"fontsize": 10, "fontname" :"Helvetica"})
            if col == 0:
                ax.set_ylabel(row2yaxis[row], fontdict={"fontsize": 10, "fontname" :"Helvetica"})

            ax.set_xticks([0, 1000000, 2000000])
            ax.set_yticks([0, 0.5, 1])
            ax.set_ylim(0, 1)
            ax.set_xlim(min(x), 2000000)
            ax.xaxis.set_major_formatter(formatter)

            if row == 0:
                ax.set_title(env2title[env_type], fontsize=14, fontname="Helvetica")
            if row == 0 and col == 0:
                ax.legend(loc='upper center', bbox_to_anchor=(1.8, 2.1), ncol=3, prop = {"size": 12})

        except Exception as error:
            print(error)

for row in range(row_nbr):
    for col, env_name in enumerate(["normal", "noise", "missing"]):
        draw_graph(env_name, axes[row, col], row, col)
        
plt.grid(False)
plt.savefig(f"./saved_plots/cheetah.png", dpi=600, bbox_inches="tight")
plt.show()

