# environment setup

from psltl.baseline_algo.crm.rl_agents.ddpg.defaults import half_cheetah_environment
# ddpg learn
from psltl.baseline_algo.crm.rl_agents.ddpg.ddpg import ddpg_learn
# vetorized env
from baselines.common.vec_env import VecNormalize
# tensorflow session
from baselines.common.tf_util import get_session
# tensorflow
import tensorflow as tf
# cmd utils
from psltl.baseline_algo.crm.cmd_util import make_vec_env, common_arg_parser


def build_cheeta_env(
    seed: int = 0,
    use_rs: bool = False):
    # defaults args
    args = ['run.py', '--alg=ddpg', '--env=Half-Cheetah-RM2-v0']
    use_rs_args = "--use_rs"    
    if use_rs:
        args.append(use_rs_args)
    
    arg_parser = common_arg_parser()
    args, _ = arg_parser.parse_known_args(args)
    config = tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=1,
                               inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    get_session(config=config)
    # for cheetah env
    env_type = "half_cheetah_environment"
    env_id = "Half-Cheetah-RM2-v0"

    env = make_vec_env(env_id, env_type, args.num_env or 1, seed, args, reward_scale=args.reward_scale, flatten_dict_observations=True)
    
    if env_type == 'mujoco':
        env = VecNormalize(env, use_tf=True)

    return env

total_timesteps = int(1e4)
env = build_cheeta_env()
config = half_cheetah_environment()


ddpg_learn(env=env, total_timesteps=total_timesteps, **config)