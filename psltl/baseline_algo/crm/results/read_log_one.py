import numpy as np
import sys
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import rolling_window

env_name = sys.argv[1]
alg = sys.argv[2]
run_nbr = sys.argv[3]

with np.load("./" + env_name + "_results" + "/" + alg + "/" + str(run_nbr) + ".npz") as data:
    successes = data["successes"]
    partial_successes = data["partial_successes"]
    rewards = data["results"]
    s = np.mean(successes, axis=1)
    ps = np.mean(partial_successes, axis=1)
    ep_lenghts = np.mean(data["ep_lengths"], axis=1)
    r = np.mean(rewards, axis=1)

org_s = s
s = rolling_window(org_s, window=20)
s = np.mean(s, axis=1)

for i in range(20, 1, -1):
    s = np.append(s, np.mean(org_s[-i:]))

org_ps = ps
ps = rolling_window(org_ps, window=20)
ps = np.mean(ps, axis=1)

for i in range(20, 1, -1):
    ps = np.append(s, np.mean(org_ps[-i:]))


fig, axes = plt.subplots(2)
 
color = 'tab:red'
axes[0].set_xlabel('Training Steps')
axes[0].set_ylabel('Success Rate', color = color)
axes[0].plot(s, color = color)
axes[0].tick_params(axis ='y', labelcolor = color)
 
# Adding Twin Axes to plot using dataset_2
ax02 = axes[0].twinx()
 
color = 'tab:green'
ax02.set_ylabel('Episode Lengths', color = color)
ax02.plot(ep_lenghts, color = color)
ax02.tick_params(axis ='y', labelcolor = color)

axes[1].set_xlabel('Training Steps')
axes[1].set_ylabel('Reward', color = color)
axes[1].plot(r, color = color)
axes[1].tick_params(axis ='y', labelcolor = color)


# plt.plot(s, label="success")
# plt.plot(ps, label="partial success")
# plt.plot(ep_lenghts, label="episode length")
plt.title(alg)
plt.legend()
plt.show()
