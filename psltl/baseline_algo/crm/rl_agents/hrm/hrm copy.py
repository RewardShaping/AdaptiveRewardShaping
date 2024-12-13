"""
Q-Learning based method
"""

import os
import random
import numpy as np
import pickle5 as pickle
from baselines import logger

def add_state_if_needed(Q,s,actions,q_init):
    if s not in Q:
        Q[s] = dict([(a,q_init) for a in actions])

def get_qmax(Q,s,actions,q_init):
    add_state_if_needed(Q,s,actions,q_init)
    return max(Q[s].values())

def get_best_action(Q,s,actions,q_init):
    qmax = get_qmax(Q,s,actions,q_init)
    best = [a for a in actions if Q[s][a] == qmax]
    return random.choice(best)

def eval_model(Q_options, Q_controller, eval_env, noise_level, env_name, gamma, q_init, eval_nbr = 5):
    actions = list(range(eval_env.action_space.n))
    temp_eval_success = []
    temp_eval_reward = []
    temp_eval_partial_success = []
    temp_eval_epi_length = []
    total_eval_reward = 0
    for _ in range(eval_nbr):
        eval_s = tuple(eval_env.reset())
        eval_success = 0
        eval_step = 0
        eval_done = False
        prev_rm_state = 0
        eval_episode_reward = 0
        option_s     = None # State where the option initiated
        option_id    = None # Id of the current option being executed
        while not eval_done:
            if option_id is None:
                valid_options = eval_env.get_valid_options()
                option_s    = eval_s
                augmented_state = tuple(eval_env.get_option_observation(option_id))
                add_state_if_needed(Q_controller, augmented_state, valid_options, q_init)
                # add_state_if_needed(Q_controller, option_s, valid_options, q_init)
                
                # print(augmented_state)
                # exit()
                option_id   =  get_best_action(Q_controller, augmented_state, valid_options, q_init)
                # option_id   =  get_best_action(Q_controller, option_s, valid_options, q_init)
                option_rews = []
            # augmented_state = tuple(eval_env.get_option_observation(option_id))
            # a = get_best_action(Q_options, augmented_state, actions,q_init)
            a = get_best_action(Q_options, eval_s, actions,q_init)
            if np.random.rand() < float(noise_level):
                # up or down 
                if a in [0, 2]:
                    a = np.random.choice([1, 3])
                # right or left
                elif a in [1, 3]:
                    a = np.random.choice([0, 2])

            eval_s, org_eval_r, eval_done, eval_info = eval_env.step(a)
            eval_s = tuple(eval_s)
            eval_r = 0.
            label = eval_info["label"]
            rm_state = eval_info["rm_state"]
            u1 = prev_rm_state
            u2 = rm_state

            if eval_env.did_option_terminate(option_id):
                option_id = None

            ####################################################
            # For parital reward evaluation!
            ####################################################
            if env_name == "taxi":
                # rm state 0: initial state
                # rm state 1: get passenger + alpha
                # rm state 2: passenger + reach destination
                # rm state 3: drop off the passenger to the destination
                if u1 == 0 and u2 == 1:
                    eval_r = 1.
                elif u1 == 1 and u2 == 2:
                    eval_r = 1.
                elif u1 == 0 and u2 == 2: 
                    eval_r = 2. # two step simultaneuosly, in taxi, this is possible becasue sometimes, taxi starts with having passeneger inside
                elif u1 == 2 and u2 ==-1:
                    eval_r = 1.

            elif env_name == "office":
                # rm state 0: initial state
                # rm state 2: get mail
                # rm state 3: get coffee
                # rm state 4: get coffee and mail
                # rm state -1: goal state
                # rm state 5: trapping state
                if u1 == 0 and u2 == 2:
                    eval_r = 1.
                elif u1 == 1 and u2 == 3:
                    eval_r = 1.
                elif u1 == 2 and u2 == 4: 
                    eval_r = 1. 
                elif u1 == 3 and u2 == 4: 
                    eval_r = 1. 
                elif u1 == 4 and u2 == -1:
                    eval_r = 1.
                elif u1 == 0 and u2 == 3:
                    eval_r = 1.

            eval_episode_reward += (gamma ** eval_step) * eval_r
            eval_step += 1
            prev_rm_state = rm_state

            if u2 == -1:
                eval_success = 1

        total_eval_reward += eval_episode_reward
        temp_eval_reward.append(eval_episode_reward)
        temp_eval_success.append(eval_success)
        temp_eval_epi_length.append(eval_step)
        temp_eval_partial_success.append(rm_state)
    
    return temp_eval_reward, temp_eval_success, temp_eval_epi_length, temp_eval_partial_success


def learn(env,
          eval_env,
          network=None,
          seed=None,
          lr=0.1,
          total_timesteps=100000,
          epsilon=0.1,
          print_freq=10000,
          gamma=0.95,
          q_init=2.0,
          hrm_lr=0.1,
          use_rs=False,
          eval_freq=1000,
          eval_nbr=5, 
          env_name: str = "office",
          missing=False,
          noise_level: float = 0.,
          test: bool = False,
          load_path: str = None,
          **others):
    """Train a tabular HRM method.

    Parameters
    -------
    env: gym.Env
        environment to train on
    network: string or a function
        This is just a placeholder to be consistent with the openai-baselines interface, but we don't really use state-approximation in tabular q-learning
    seed: int or None
        prng seed. The runs with the same seed "should" give the same results. If None, no seeding is used.
    lr: float
        learning rate
    total_timesteps: int
        number of env steps to optimizer for
    epsilon: float
        epsilon-greedy exploration
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    gamma: float
        discount factor
    q_init: float
        initial q-value for unseen states
    hrm_lr: float
        learning rate for the macro-controller
    use_rs: bool
        use reward shaping
    """

    # Running Q-Learning
    successes = []
    partial_successes = []
    episode_lengths = []
    episode_rewards = []
    # Running Q-Learning
    step         = 0
    num_episodes = 0
    reward_total = 0
    actions      = list(range(env.action_space.n))
    Q_controller = {}   # Q-values for the meta-controller
    Q_options    = {}   # Q-values for the option policies
    option_s     = None # State where the option initiated
    option_id    = None # Id of the current option being executed
    option_rews  = []   # Rewards obtained by the current option
    while step < total_timesteps:
        s = tuple(env.reset())
        while True:
            # Selecting an option if needed
            if option_id is None:
                valid_options = env.get_valid_options()
                option_s    = s
                option_s = tuple(eval_env.get_option_observation(option_id))
                add_state_if_needed(Q_controller, option_s, valid_options, q_init)
                option_id   = random.choice(valid_options) if random.random() < epsilon else get_best_action(Q_controller, s, valid_options,q_init)
                option_rews = []

            # Selecting and executing an action
            if random.random() < epsilon:
                a = random.choice(actions)
            else: 
                a = get_best_action(Q_options, s, actions,q_init)
                
            # 0: up, 1: right, 2: down, 3: left
            if np.random.rand() < float(noise_level):
                # up or down 
                if a in [0, 2]:
                    a = np.random.choice([1, 3])
                # right or left
                elif a in [1, 3]:
                    a = np.random.choice([0, 2])
            sn, r, done, info = env.step(a)
            sn = tuple(sn)

            # Saving the real reward that the option is getting
            if use_rs:
                option_rews.append(info["rs-reward"])
            else:
                option_rews.append(r)

            # Updating the option policies
            for _s,_a,_r,_sn,_done in env.get_experience():
                _s,_sn = tuple(_s), tuple(_sn)
                add_state_if_needed(Q_options, _s, actions, q_init)
                if _done: _delta = _r - Q_options[_s][_a]
                else:     _delta = _r + gamma * get_qmax(Q_options,_sn,actions,q_init) - Q_options[_s][_a]
                Q_options[_s][_a] += lr*_delta

            # Updating the meta-controller if needed 
            # Note that this condition always hold if done is True
            if env.did_option_terminate(option_id):
                option_sn = sn
                option_reward = sum([_r*gamma**_i for _i,_r in enumerate(option_rews)])
                if done:
                    _delta = option_reward - Q_controller[option_s][option_id]
                else:    
                    q_max = get_qmax(Q_controller, option_sn, env.get_valid_options(), q_init)
                    _delta = option_reward + gamma ** (len(option_rews)) * q_max - Q_controller[option_s][option_id]
                Q_controller[option_s][option_id] += hrm_lr*_delta
                option_id = None

            # Moving to the next state
            reward_total += r
            step += 1
            # if step%print_freq == 0:
            #     logger.record_tabular("steps", step)
            #     logger.record_tabular("episodes", num_episodes)
            #     logger.record_tabular("total reward", reward_total)
            #     logger.dump_tabular()
            #     reward_total = 0

            if step % eval_freq == 0:
                temp_eval_reward, temp_eval_success, temp_eval_epi_length, temp_eval_partial_success \
                    = eval_model(Q_options, Q_controller, eval_env, noise_level, env_name, gamma, q_init, eval_nbr)
                print("step: {} and reward: {}".format(step, np.mean(temp_eval_reward)))
                episode_rewards.append(sum(temp_eval_reward) / len(temp_eval_reward))
                episode_lengths.append(temp_eval_epi_length)
                successes.append(temp_eval_success)
                partial_successes.append(temp_eval_partial_success)
                print(np.mean(temp_eval_epi_length))
        
            if done:
                num_episodes += 1
                break
            s = sn
    
    if use_rs:
        save_path = "./results/" + env_name + "/hrm_rs"
    elif not use_rs:
        save_path = "./results/" + env_name + "/hrm"
    
    if bool(missing):
        save_path += "_missing"

    if noise_level > 0:
        save_path += "_noise_" + str(noise_level)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # with open(save_path + "/model" + str(seed) + ".pkl", "wb") as f:
    #     pickle.dump(Q, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    np.savez(
        save_path + "/" + str(seed),
        successes=successes,
        partial_successes=partial_successes,
        results=episode_rewards,
        ep_lengths=episode_lengths
            )
