from psltl.learner.learner import Learner
from psltl.utils.parser_info import get_parser
from psltl.utils.param_info import get_param
from psltl.utils.utils import set_seed

# In order to load the saved map, pickle require us to call the following,
from psltl.envs.common.cont.water.water_world import BallAgent, Ball


args = get_parser()

alg = args.algo_name
env_name = args.env_name

assert env_name in ["office", "water", "cheetah", "toy", "taxi"], print("Env types " + str(env_name) + " hasn't been defined yet.")
assert alg in ["dqn", "ddqn", "ddpg", "ppo", "a2c", "td3", "sac"]

match_reward_types = {"p": "progress", "h": "hybrid", "n": "naive"}

reward_type = match_reward_types[args.reward_types]
if float(args.noise_level) > 0:
    env_type = "noise"
elif bool(args.missing):
    env_type = "missing"
else:
    env_type = "normal"

params = get_param(env_name, reward_type, alg, env_type)

# Total run and timesteps for each run
params.total_run = int(args.total_run)
# Environment setup
params.episode_step = int(args.episode_step) # how many steps per episode, terminal condition for each episode
# make map size = map id
params.map_size = int(args.map_id)
params.map_id = int(args.map_id)
params.seed = int(args.seed)
params.violation_end = bool(args.violation_end)
params.env_name = env_name
params.missing = args.missing
params.algo_name = args.algo_name
params.noise_level = float(args.noise_level)
params.human = True if str(args.human) == "True" else False
params.use_adrs = bool(args.use_adrs)
params.reward_types = reward_type
params.version = int(args.version)
# rolling window size for evaluation
params.rolling = int(args.rolling) 

if not bool(args.default_setting):
    params.node_embedding = args.node_embedding
    params.use_one_hot = args.use_one_hot # one hot encoding 
    params.total_timesteps = int(args.total_timesteps)
    params.eval_freq = int(args.eval_freq)
    # Training setup for algorithm
    params.gamma = float(args.gamma)
    params.train_freq = int(args.train_freq)    
    params.target_update_interval = int(args.update_interval)
    params.learning_starts = int(args.learning_starts)
    params.max_grad_norm = float(args.max_grad_norm)
    params.buffer_size = int(args.buffer_size)
    params.batch_size = int(args.batch_size)

    # adaptive reward shaping
    if float(args.theta) < 0:
        params.theta = "dist"
    else:
        params.theta = float(args.theta)

    if reward_type == "naive":
        # for the naive reward function, we do not use adaptive reward shaping method
        params.theta = 0

    exp_init = float(args.exp_init)
    exp_final = float(args.exp_final)
    exp_fraction = float(args.exp_fraction)
    params.adrs_update = int(args.adrs_update) 
    # Exploration setup
    params.exploration_initial_eps = exp_init
    params.exploration_final_eps = exp_final
    params.exploration_fraction = exp_fraction
    # Learning rate setup
    params.learning_rate_start = float(args.lr_start)
    params.learning_rate_end = float(args.lr_end)
    params.learning_fraction = float(args.lr_fraction)
    # Reward function setup
    params.hybrid_eta = float(args.hybrid_eta) # (1-eta) * progress + eta * distance 

assert (params.node_embedding and params.use_one_hot) == False, "only one of representation must be used, one hot or node embedding"

set_seed(params.seed)
if args.hp_tuning:
    from psltl.hptunning.hp_objective import hp_tunning
    hp_tunning(args, params)
else:
    leaner = Learner(params)
    leaner.learn()

# python run.py --env_name taxi --total_timesteps 10000 --total_run 1 --reward_types p --default_setting True --seed 100 --algo_name dqn --adrs_update 25 --use_adrs True --node_embedding True --eval_freq 100