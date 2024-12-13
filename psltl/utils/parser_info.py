import argparse


def get_parser():
    env_types = ["office", "water", "cheetah", "toy", "taxi"]

    parser = argparse.ArgumentParser(prog="run_experiments", description='Runs RL experiment over a particular environment using reward shaping based on LTL')
    
    # reward function setup
    parser.add_argument('--reward_types', default="p", type=str,
                        help='This parameter indicated reward types. The options are: p, d, h, which means p: progress, d: distance, h: hybrid')
    parser.add_argument('--use_adrs', default=False, type=bool,
                        help='This parameter indicated using adaptive reward shaping or not')
    parser.add_argument('--hybrid_eta', default=0.005, type=float,
                        help='This parameter indicated adaptive reward shaping gamma factor for trade offs between distance and progress')
    parser.add_argument('--adrs_update', default=500, type=int,
                        help='This parameter indicated update step for distance function adaptive reward shaping')
    parser.add_argument('--adrs_mu', default=0.5, type=float,
                        help='adrs mu for ADRS')
    parser.add_argument('--human', default=False, type=bool,
                        help='reward desgined by human e.x. extrinc reward')
    parser.add_argument('--version', default=1, type=int,
                        help='different type of reward function update 0: update reward function with trajectories 1: update reward function with the best..')
    parser.add_argument('--theta', default=-1, type=int,
                        help='theta for adrs update')
    
    # environment setup
    parser.add_argument('--env_name', default='office', type=str,
                        help='This parameter indicated which env types to solve. The options are: ' + str(env_types))
    parser.add_argument('--default_setting', default=True, type=bool,
                        help='use default setting')
    parser.add_argument('--episode_step', default=int(1e2), type=int,
                        help='This parameter indicated how many steps are allowed in one episode')
    parser.add_argument('--noise_level', default=0, type=float,
                        help='noise level for action')
    parser.add_argument('--gamma', default=0.9, type=float,
                        help='This parameter indicated MDP discount factor')
    parser.add_argument('--violation_end', default=False, type=bool,
                        help='Environment will end if the designed specification is violated')
    parser.add_argument('--missing', default=False, type=bool,
                        help='Environment will have no goal state')
    parser.add_argument('--map_size', default=1, type=int,
                        help='This parameter indicated especially for office world, increase map size')
    parser.add_argument('--map_id', default=1, type=int,
                        help='This parameter will be used to decide which map we will use ')
    
    # evaluation setup
    parser.add_argument('--eval_freq', default=100, type=int,
                        help='This parameter indicated evaluation frequncy')
    parser.add_argument('--seed', default=42, type=int,
                        help='default seed value')
    parser.add_argument('--rolling', default=20, type=int,
                        help='rolling window for evaluation')
    
    # RL training setup
    parser.add_argument('--algo_name', default="dqn", type=str,
                        help='This parameter indicated which algorithm for RL we will use')
    parser.add_argument('--total_timesteps', default=int(1e4), type=int,
                        help='This parameter indicated total training time steps')
    parser.add_argument('--total_run', default=3, type=int,
                        help='This parameter indicated how many times we will run each environment with different seeds')
    parser.add_argument('--use_one_hot', default=False, type=bool,
                        help='This parameter indicated whether we will use one hot encoding or not')
    parser.add_argument('--node_embedding', default=False, type=bool,
                        help='This parameter indicated whether we will use qrm like representation')
    parser.add_argument('--lr_start', default=0.1, type=float,
                        help='This parameter will learning rate start')
    parser.add_argument('--lr_end', default=0.0001, type=float,
                        help='This parameter will learnint rate end')
    parser.add_argument('--lr_fraction', default=0.4, type=float,
                        help='This parameter will learning rate fraction')
    parser.add_argument('--exp_init', default=0.1, type=float,
                        help='This parameter will exploration rate start')
    parser.add_argument('--exp_final', default=0.05, type=float,
                        help='This parameter will exploration rate end')
    parser.add_argument('--exp_fraction', default=0.4, type=float,
                        help='This parameter will exploration fraction')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='batch size')
    parser.add_argument('--buffer_size', default=32, type=int,
                        help='buffer size')
    parser.add_argument('--train_freq', default=1, type=int,
                        help='train_freq')
    parser.add_argument('--gradient_steps', default=1, type=int,
                        help='gradient steps')
    parser.add_argument('--update_interval', default=100, type=int,
                        help='update interval')
    parser.add_argument('--learning_starts', default=1, type=int,
                        help='when we start to train algorithm for off-policy methods')
    parser.add_argument('--max_grad_norm', default=1., type=float,
                        help='maximum value for gradient norm')
    
    # hyperparam tuning
    parser.add_argument('--hp_tuning', default=False, type=bool,
                        help='hyperparam tuning')
    args = parser.parse_args()

    return args

    
