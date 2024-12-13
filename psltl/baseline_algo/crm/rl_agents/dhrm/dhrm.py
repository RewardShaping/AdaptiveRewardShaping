import os
import tempfile

import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np

import baselines.common.tf_util as U
from baselines.common.tf_util import load_variables, save_variables
from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines.common import set_global_seeds

from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.utils import ObservationInput

from baselines.common.tf_util import get_session
from baselines.deepq.models import build_q_func

from psltl.baseline_algo.crm.rl_agents.dhrm.options import OptionDQN, OptionDDPG
from psltl.baseline_algo.crm.rl_agents.dhrm.controller import ControllerDQN


def eval_model(options, controller, eval_env, noise_level, env_name, gamma, eval_nbr = 5):
    temp_eval_success = []
    temp_eval_reward = []
    temp_eval_partial_success = []
    temp_eval_epi_length = []
    for _ in range(eval_nbr):
        option_s    = None # State where the option initiated
        option_id   = None # Id of the current option being executed

        obs = eval_env.reset()
        options.reset()
        reset = True
        
        total_eval_reward = 0
        eval_success = 0
        eval_step = 0
        eval_done = False
        # for water world case, initial state is 1, and success state is 0
        eval_episode_reward = 0
        prev_rm_state = 1
        while not eval_done:
            # Selecting an option if needed
            if option_id is None:
                valid_options = eval_env.get_valid_options()
                option_s    = obs
                option_id   = controller.get_action(option_s, valid_options)

            action = options.get_action(eval_env.get_option_observation(option_id), eval_step, reset)
            reset = False

            if env_name == "water":
                if np.random.rand() < float(noise_level):
                    # up or down 
                    if action in [0, 2]:
                        action = np.random.choice([1, 3])
                    # right or left
                    elif action in [1, 3]:
                        action = np.random.choice([0, 2])
            else:
                if np.random.rand() < float(noise_level):
                    action += np.random.uniform(-0.1, 0.1)
                    action = np.clip(action, -1., 1.)

            eval_obs, eval_r, eval_done, eval_info = eval_env.step(action)
            eval_r = 0.

            
            rm_state = eval_info["rm_state"]
            # else:
            #     rm_state = eval_info[0]["rm_state"]

            u1 = prev_rm_state
            u2 = rm_state
            if env_name == "water":
                # rm state 1: initial state
                # rm state 2: red strict
                # rm state 3: green strict
                # rm state -1: touch blue -> success 
                if u1 == 1 and u2 == 2:
                    eval_r = 1.
                elif u1 == 2 and u2 == 3:
                    eval_r = 1.
                elif u1 == 3 and u2 == -1:
                    eval_r = 1.
                # if u2 == -2:
                #     eval_r = -10.
                if u2 == -1:
                    eval_success = 1

            elif env_name == "cheetah":
                if u1 == 0 and u2 == 1:
                    eval_r = 1.
                elif u1 == 1 and u2 == 2:
                    eval_r = 1.
                elif u1 == 2 and u2 == 3:
                    eval_r = 1.
                elif u1 == 3 and u2 == 4:
                    eval_r = 1.
                elif u1 == 4 and (u2 == -1 or u2 == 5):
                    eval_r = 1.
                if u2 == -1 or u2 == 5:
                    eval_success = 1

            if eval_env.did_option_terminate(option_id):
                valid_options = [] if eval_done else eval_env.get_valid_options()
                option_id = None

            obs = eval_obs
            eval_episode_reward += gamma ** eval_step * eval_r
            eval_step += 1
            prev_rm_state = rm_state

        total_eval_reward += eval_episode_reward
        temp_eval_reward.append(eval_episode_reward)
        temp_eval_success.append(eval_success)
        temp_eval_epi_length.append(eval_step)
        temp_eval_partial_success.append(rm_state)
        
    return temp_eval_success, temp_eval_reward, temp_eval_partial_success, temp_eval_epi_length


def learn(env,
          eval_env,
          use_ddpg=False,
          gamma=0.9,
          use_rs=False,
          controller_kargs={},
          option_kargs={},
          seed=None,
          total_timesteps=100000,
          print_freq=100,
          callback=None,
          checkpoint_path=None,
          checkpoint_freq=10000,
          load_path=None,
          eval_freq=1000,
          eval_nbr=5,
          env_name="water",
          missing=False,
          noise_level:float = 0.,
          test: bool = False,
          **others):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    use_ddpg: bool
        whether to use DDPG or DQN to learn the option's policies
    gamma: float
        discount factor
    use_rs: bool
        use reward shaping
    controller_kargs
        arguments for learning the controller policy.
    option_kargs
        arguments for learning the option policies.
    seed: int or None
        prng seed. The runs with the same seed "should" give the same results. If None, no seeding is used.
    total_timesteps: int
        number of env steps to optimizer for
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    load_path: str
        path to load the model from. (default: None)

    Returns
    -------
    act: ActWrapper (meta-controller)
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    act: ActWrapper (option policies)
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    # Create all the functions necessary to train the model

    sess = get_session(config=tf.ConfigProto(inter_op_parallelism_threads=1,
                   intra_op_parallelism_threads=1))
    set_global_seeds(seed)

    controller  = ControllerDQN(env, **controller_kargs)
    if use_ddpg:
        options = OptionDDPG(env, gamma, total_timesteps, **option_kargs)
    else:
        options = OptionDQN(env, gamma, total_timesteps, **option_kargs)

    option_s    = None # State where the option initiated
    option_id   = None # Id of the current option being executed
    option_rews = []   # Rewards obtained by the current option

    episode_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset()
    options.reset()
    reset = True
    
    successes = []
    partial_successes = []
    episode_lengths = []
    my_episode_rewards = []
    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_path or td

        model_file = os.path.join(td, "model")
        model_saved = False

        if tf.train.latest_checkpoint(td) is not None:
            load_variables(model_file)
            logger.log('Loaded model from {}'.format(model_file))
            model_saved = True

        elif load_path is not None:
            load_variables(load_path)
            logger.log('Loaded model from {}'.format(load_path))


        for t in range(total_timesteps):
            if callback is not None:
                if callback(locals(), globals()):
                    break

            # Selecting an option if needed
            if option_id is None:
                valid_options = env.get_valid_options()
                option_s    = obs
                option_id   = controller.get_action(option_s, valid_options)
                option_rews = []

            # Take action and update exploration to the newest value
            action = options.get_action(env.get_option_observation(option_id), t, reset)
            reset = False
            new_obs, rew, done, info = env.step(action)

            # Saving the real reward that the option is getting
            if use_rs:
                option_rews.append(info["rs-reward"])
            else:
                option_rews.append(rew)

            # Store transition for the option policies
            for _s,_a,_r,_sn,_done in env.get_experience():
                options.add_experience(_s,_a,_r,_sn,_done)

            # Learn and update the target networks if needed for the option policies
            options.learn(t)
            options.update_target_network(t)

            # Update the meta-controller if needed 
            # Note that this condition always hold if done is True
            if env.did_option_terminate(option_id):
                option_sn = new_obs
                option_reward = sum([_r*gamma**_i for _i,_r in enumerate(option_rews)])
                valid_options = [] if done else env.get_valid_options()
                controller.add_experience(option_s, option_id, option_reward, option_sn, done, valid_options,gamma**(len(option_rews)))
                controller.learn()
                controller.update_target_network()
                controller.increase_step()
                option_id = None

            obs = new_obs
            episode_rewards[-1] += rew

            if done:
                obs = env.reset()
                options.reset()
                episode_rewards.append(0.0)
                reset = True

            # General stats
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            num_episodes = len(episode_rewards)
            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.dump_tabular()
            
            if t % eval_freq == 0:
                temp_eval_success, temp_eval_reward, temp_eval_partial_success, temp_eval_epi_length = eval_model(options, controller, eval_env, noise_level, env_name, gamma, eval_nbr)
                
                total_eval_reward = sum(temp_eval_reward) / len(temp_eval_reward)
                print("step: {} and reward: {} and success: {}".format(t, total_eval_reward, np.mean(temp_eval_success)))
                my_episode_rewards.append(temp_eval_reward)
                episode_lengths.append(temp_eval_epi_length)
                successes.append(temp_eval_success)
                partial_successes.append(temp_eval_partial_success)

            # if (checkpoint_freq is not None and
            #         num_episodes > 100 and t % checkpoint_freq == 0):
            #     if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
            #         if print_freq is not None:
            #             logger.log("Saving model due to mean reward increase: {} -> {}".format(
            #                        saved_mean_reward, mean_100ep_reward))
            #         save_variables(model_file)
            #         model_saved = True
            #         saved_mean_reward = mean_100ep_reward
                

        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            #load_variables(model_file)
                
        if use_rs:
            save_path = os.getcwd() + "/results/" + env_name + "/hrm_rs"
        elif not use_rs:
            save_path = os.getcwd() + "/results/" + env_name + "/hrm_rs"

        # act.save_act()
        
        if bool(missing):
            save_path += "_missing"

        if noise_level > 0:
            save_path += "_noise_" + str(noise_level)

        fd_path = save_path

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # act.save_act(fd_path, fd_path + "/model" + str(seed) + ".pkl")
        print("save path is", save_path)
        print("saving happening")
        print("successe", successes)
        np.savez(
            save_path + "/" + str(seed),
            successes=successes,
            partial_successes=partial_successes,
            results=my_episode_rewards,
            ep_lengths=episode_lengths
                )
        

    return controller.act, options.act
