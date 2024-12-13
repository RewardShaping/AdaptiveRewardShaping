import subprocess
from itertools import product
import sys

# python run.py --alg=hrm --env=Taxi-v0 --num_timesteps=5e5 --gamma=0.9 --env_name="taxi" --seed 0
# python run.py --alg=dhrm --env=Water-single-M3-v0 --num_timesteps=2e6 --gamma=0.9 --env_name="water" --use_rs --seed 0
# python run.py --alg=dhrm --num_timesteps=2e6 --gamma=0.99 --use_crm --env=Half-Cheetah-RM2-v0 --normalize_observations=True --seed=0 --env_name=cheetah

algs = ["hrm"]
# grid worlds
envs = ["Taxi-v0", "Office-single-v0"]
envs2names = {"Taxi-v0": "taxi", "Office-single-v0": "office"}
names2steps = {"taxi": 5e5, "office": 6e4}
names2gamma = {"taxi": 0.9, "office": 0.95}
env_types = ["complete", "noise", "missing"]
# continous worlds
# envs = ["Water-single-M3-v0", "Half-Cheetah-RM2-v0"]
envs = ["Water-single-M3-v0"]
envs2names = {"Water-single-M3-v0": "water", "Half-Cheetah-RM2-v0": "cheetah"}
names2steps = {"water": 2e6, "cheetah": 2e6}
names2gamma = {"water": 0.9, "cheetah": 0.99}
env_types = ["complete", "noise", "missing"]
env_types = ["complete"]

# env_names = ["office", "taxi", "cheetah"]
# env_names = ["office", "taxi", "water", "cheetah"]

total_comb = list(
            product([i for i in algs], [i for i in envs], [i for i in env_types])
            )

with open("temp_experiments.txt", "w") as f:
    f.write("ArrayTaskID,AlgoName,Env,TimeSteps,Gamma,EnvName,Seed,EnvType\n")
    idx = 0
    for element in total_comb:
        env = element[1]
        name = envs2names[env]
        if name in ["office", "taxi", "water", "cheetah"]:
            for seed in range(10):
                idx += 1
                steps = names2steps[name]
                gamma = names2gamma[name]
                env_type = element[-1]
                f.write(f"{idx},{element[0]},{env},{steps},{gamma},{name},{seed},{env_type}\n")

command = ['awk', 'BEGIN {FS=OFS=","} {gsub(/,/, "\t"); print}', 'temp_experiments.txt']
output_file = "hrm_experiments.txt"
subprocess.run(command, stdout=open(output_file, 'w'))
del_command = ['rm', 'temp_experiments.txt']
subprocess.run(del_command, stdout=subprocess.PIPE)