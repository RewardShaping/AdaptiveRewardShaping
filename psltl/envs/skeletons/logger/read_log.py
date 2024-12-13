import pandas as pd
import matplotlib.pyplot as plt

load_path = "/u/hbt9su/PartialSatLTL/psltl/envs/skeletons/logger/hltl_test/progress.csv"

result = pd.read_csv(load_path)


mr = "eval/mean_reward"
sr = "eval/recent_success_rate"


mean_reward = list(result[mr])
mean_sr = list(result[sr])

plt.subplot(2, 1, 1)
plt.plot(mean_reward)
plt.title('Plot 1: reward')

plt.subplot(2, 1, 2)
plt.plot(mean_sr)
plt.title('Plot 2: success rate')

# Adjust layout for better spacing
plt.tight_layout()

# Show the plots
plt.savefig("./result.png", dpi=600)