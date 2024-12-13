"""
Q-Learning based method
"""
import os
import numpy as np
import random, time
import copy
from baselines import logger
import pickle5 as pickle

def eval_model(Q, eval_env, noise_level, env_name, gamma, q_init, eval_nbr = 5):
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
        while not eval_done:
            a = get_best_action(Q,eval_s,actions,q_init)
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


def get_qmax(Q,s,actions,q_init):
    if s not in Q:
        Q[s] = dict([(a,q_init) for a in actions])
    return max(Q[s].values())

def get_best_action(Q,s,actions,q_init):
    qmax = get_qmax(Q,s,actions,q_init)
    best = [a for a in actions if Q[s][a] == qmax]
    return random.choice(best)

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
          use_crm=False,
          use_rs=False,
          eval_freq=1000,
          eval_nbr=5, 
          env_name: str = "office",
          missing=False,
          noise_level: float = 0.,
          test: bool = False,
          load_path: str = None
          ):
    """Train a tabular q-learning model.

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
    use_crm: bool
        use counterfactual experience to train the policy
    use_rs: bool
        use reward shaping
    """
    if test:
        assert load_path != None, "if you want to test model, load path should be defined!"
        with open(load_path, 'rb') as handle:
            Q = pickle.load(handle)
        
        temp_eval_reward, temp_eval_success, _, _ = eval_model(Q, eval_env, noise_level, env_name, gamma, q_init, eval_nbr)
        print(temp_eval_reward)
        print(temp_eval_success)
        # python run.py --alg=deepq --env=Water-single-M3-v0 --num_timesteps=2e3 --gamma=0.9 --env_name="water" --use_crm --seed 0 --test=True --load_path="/home/mj/Projects/PartialSatLTL/psltl/baseline_algo/crm/results/water/crm/model0.pkl"
        exit()


    successes = []
    partial_successes = []
    episode_lengths = []
    episode_rewards = []
    # Running Q-Learning
    reward_total = 0
    step = 0
    num_episodes = 0
    Q = {}
    actions = list(range(env.action_space.n))
    eval_env.reset()
    
    while step < total_timesteps:
        s = tuple(env.reset())
        if s not in Q: Q[s] = dict([(a,q_init) for a in actions])
        while True:
            # Selecting and executing the action
            a = random.choice(actions) if random.random() < epsilon else get_best_action(Q,s,actions,q_init)
            
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
            # Updating the q-values
            
            experiences = []
            if use_crm:
                # Adding counterfactual experience (this will alrady include shaped rewards if use_rs=True)
                for _s,_a,_r,_sn,_done in info["crm-experience"]:
                    experiences.append((tuple(_s),_a,_r,tuple(_sn),_done))
            elif use_rs:
                # Include only the current experince but shape the reward
                experiences = [(s,a,info["rs-reward"],sn,done)]
            else:
                # Include only the current experience (standard q-learning)
                experiences = [(s,a,r,sn,done)]

            for _s,_a,_r,_sn,_done in experiences:
                if _s not in Q: Q[_s] = dict([(b,q_init) for b in actions])
                if _done: _delta = _r - Q[_s][_a]
                else:     _delta = _r + gamma*get_qmax(Q,_sn,actions,q_init) - Q[_s][_a]
                Q[_s][_a] += lr*_delta

            # moving to the next state
            reward_total += r
            step += 1
            # if step%print_freq == 0:
            #     logger.record_tabular("steps", step)
            #     logger.record_tabular("episodes", num_episodes)
            #     logger.record_tabular("total reward", reward_total)
            #     logger.dump_tabular()
            #     reward_total = 0
            
            if step % eval_freq == 0:
                temp_eval_reward, temp_eval_success, temp_eval_epi_length, temp_eval_partial_success = eval_model(Q, eval_env, noise_level, env_name, gamma, q_init, eval_nbr)
                print("step: {} and reward: {}".format(step, np.mean(temp_eval_reward)))
                episode_rewards.append(sum(temp_eval_reward) / len(temp_eval_reward))
                episode_lengths.append(temp_eval_epi_length)
                successes.append(temp_eval_success)
                partial_successes.append(temp_eval_partial_success)
                print(np.mean(temp_eval_epi_length))
            
            sn = tuple(sn)
            if done:
                num_episodes += 1
                break
            s = sn

    if use_crm and not use_rs:
        save_path = "./results/" + env_name + "/crm"
    elif not use_crm and use_rs:
        save_path = "./results/" + env_name + "/rs"
    elif use_crm and use_rs:
        save_path = "./results/" + env_name + "/crm_rs"
    
    if bool(missing):
        save_path += "_missing"

    if noise_level > 0:
        save_path += "_noise_" + str(noise_level)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(save_path + "/model" + str(seed) + ".pkl", "wb") as f:
        pickle.dump(Q, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    np.savez(
        save_path + "/" + str(seed),
        successes=successes,
        partial_successes=partial_successes,
        results=episode_rewards,
        ep_lengths=episode_lengths
            )