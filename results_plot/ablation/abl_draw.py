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
    print("latex can not be found")

formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 2))  # Adjust power limits as needed
env_names = ["taxi", "water"]
env_type = "missing"
env2algo = {"water": "ddqn", "taxi": "dqn"}
c_orange = (0.8666666666666667, 0.5176470588235295, 0.3215686274509804)
# c_orange = (1, 0, 1)
c_blue = (0.2980392156862745, 0.4470588235294118, 0.6901960784313725)
# c_blue = (.5, 1, 0)
c_purple = (0.5058823529411764, 0.4470588235294118, 0.7019607843137254)
c_pink = (0.5058823529411764, 0.4470588235294118, 0.7019607843137254)
# c_pink = (0, 1, 1)
alg2color = {
            "p50": (0.7686274509803922, 0.3058823529411765, 0.3215686274509804), 
            "p500": (0.8666666666666667, 0.5176470588235295, 0.3215686274509804),  
            # orangfe
            "p5000": (0.8, 0, 0.8), \
            "h50": (0.5058823529411764, 0.4470588235294118, 0.7019607843137254),  
            # purple
            "h500": (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),\
            "h5000": (0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
            # blue
            "progress_adrs": (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),\
            "hybrid_adrs": (0.2980392156862745, 0.4470588235294118, 0.6901960784313725),\
            "naive_adrs": (0.5058823529411764, 0.4470588235294118, 0.7019607843137254), 
            }

alg2color = {
            "p50": c_orange, 
            "p500": c_blue,  
            "p5000": c_pink, 
            "h50": c_orange,  
            "h500": c_blue,
            "h5000": c_pink,
            # blue
            "progress_adrs": (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),\
            "hybrid_adrs": (0.2980392156862745, 0.4470588235294118, 0.6901960784313725),\
            "naive_adrs": (0.5058823529411764, 0.4470588235294118, 0.7019607843137254), 
            }

env2step = {"office": 100, "taxi": 1000, "water": 1000, "cheetah": 1000}
env2total_step = {"office": 6e2,  "taxi": 5e2, "water": 2e3, "cheetah": 2e3}
env2title = {"normal": "Deterministic".upper(), "noise": "Noisy".upper(), "missing": "Infeasible".upper()}

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 4), gridspec_kw={'hspace': .9})
row2type = {0: "reward", 1: "reward"}
row2yaxis = {0: "Office", 1: "Taxi"}
row_nbr = 2
row2rl_algo = {0: "Office", 1: "Taxi"}

row2type = {0: "reward", 1: "reward", 2: "reward" }
row2yaxis = {0: "theta: 2000", 1: "theta: 5000", 2: "theta: 10000"}
row2yaxis = {0: "Normalized Reward", 1: "Normalized Reward"}
row_nbr = 3
count = 0
update2theta = {50: 2000, 500: 5000, 5000: 10000}
alg_names = ["progress_adrs", "hybrid_adrs"]

def draw_graph(adrs_update, alg_name, env_name, ax, row, col):
    global count
    theta = update2theta[adrs_update]
    rl_algo = env2algo[env_name]

    try:
        df = pd.read_csv(f"./{env_type}_csv/reward_{env_name}_{rl_algo}_{alg_name}_theta{theta}_update{adrs_update}.csv")
        # Sample data
        mean = df["mean"]
        std = df["std"]
        
        x = np.arange(len(mean)) * 1000 # X-axis values
        y = np.array(mean)   # Mean or median values
        std_dev = np.array(std)  # Standard deviation values
        
        # Plot the mean or median line
        # l = alg_name.upper()
        if alg_name == "progress_adrs":
            l = f"$\\theta$={theta}, $N$={adrs_update}"
            
        if alg_name == "hybrid_adrs":
            l = f"$\\theta$={theta}, $N$={adrs_update}"
            

        c = alg2color[alg_name[0] + f"{adrs_update}"]
        ax.plot(x, y, label=l, linestyle='-', color=c)
        
        ax.fill_between(x, y - std_dev, y + std_dev, alpha=0.2, color=c) # , label='Std Dev'
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add labels and legend
        ax.set_xlabel('Training Steps', fontdict={"fontsize": 10, "fontname" :"Helvetica"})
        if col == 0:
            ax.set_ylabel(row2yaxis[row], fontdict={"fontsize": 10, "fontname" :"Helvetica"})
        
        if col == 0:
            ax.set_xticks([0, 250000, 500000])
            ax.set_xlim(min(x), 500000)
        
        if col == 1:
            ax.set_xticks([0, 500000, 1000000])
            ax.set_xlim(min(x), 1000000)

        ax.set_yticks([0, 0.5, 1])
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_formatter(formatter)

        if row == 0:
            ax.set_title(f"Update Interval: {adrs_update}", fontsize=10, fontname="Helvetica")
            if env_name == "taxi":
                ax.set_title(f"TAXI", fontsize=10, fontname="Helvetica")
            if env_name == "water":
                ax.set_title(f"WATER", fontsize=10, fontname="Helvetica")
        
            if col == 1:
                ax.text(10., -.6, '(a) Adaptive progression reward', fontsize=11, ha='center')
        
        if row == 1 and col == 1:
            ax.text(-1., -.6, '(b) Adaptive hybrid reward', fontsize=11, ha='center')
        if row == 0 and col == 0:
            if count >= 2:
                pass
            else:
                count += 1
            ax.legend(loc='upper center', bbox_to_anchor=(1., 1.6), ncol=3, prop = {"size": 8})
        
    except:
        pass

adrs_updates = [50, 500, 5000]
row_nbr = 1
row = 0
for col, env_name in enumerate(env_names):
    for row, alg_name in enumerate(alg_names):
        for adrs_update in adrs_updates:
            draw_graph(adrs_update, alg_name, env_name, axes[row, col], row, col)
        
plt.grid(False)
plt.savefig(f"./saved_plots/ablation.png", dpi=600, bbox_inches="tight")
plt.show()

