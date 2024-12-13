import os
import tempfile

import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np
import copy

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


def eval_model(act, eval_env, noise_level, env_name, gamma, eval_nbr = 5):
    temp_eval_success = []
    temp_eval_reward = []
    temp_eval_partial_success = []
    temp_eval_epi_length = []
    for _ in range(eval_nbr):
        eval_s = tuple(eval_env.reset())
        total_eval_reward = 0
        eval_success = 0
        eval_step = 0
        eval_done = False
        # for water world case, initial state is 1, and success state is 0
        eval_episode_reward = 0
        prev_rm_state = 1
        
        while not eval_done:
            a = act(np.array(eval_s)[None], update_eps=0)[0]
            
            if np.random.rand() < float(noise_level):
                # up or down 
                if a in [0, 2]:
                    a = np.random.choice([1, 3])
                # right or left
                elif a in [1, 3]:
                    a = np.random.choice([0, 2])

            eval_s, eval_r, eval_done, eval_info = eval_env.step(a)
            eval_s = tuple(eval_s)
            eval_r = 0.
            rm_state = eval_info["rm_state"]
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
            eval_episode_reward += gamma ** eval_step * eval_r
            eval_step += 1
            prev_rm_state = rm_state

        total_eval_reward += eval_episode_reward
        temp_eval_reward.append(eval_episode_reward)
        temp_eval_success.append(eval_success)
        temp_eval_epi_length.append(eval_step)
        temp_eval_partial_success.append(rm_state)
        
    return temp_eval_success, temp_eval_reward, temp_eval_partial_success, temp_eval_epi_length


class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params
        self.initial_state = None

    @staticmethod
    def load_act(path):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        act = deepq.build_act(**act_params)
        sess = tf.Session()
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            load_variables(os.path.join(td, "model"))

        return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def step(self, observation, **kwargs):
        # DQN doesn't use RNNs so we ignore states and masks
        kwargs.pop('S', None)
        kwargs.pop('M', None)
        return self._act([observation], **kwargs), None, None, None

    def save_act(self, folder_path=None, path=None):
        """Save model to a pickle located at `path`"""
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print("path??", path)
        if path is None:
            path = os.path.join(logger.get_dir(), "model.pkl")

        with tempfile.TemporaryDirectory() as td:
            save_variables(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()

        print("openning path??", path)
        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params), f)

    def save(self, path):
        save_variables(path)


def load_act(path):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle

    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    return ActWrapper.load_act(path)


def learn(env,
          eval_env,
          network,
          seed=None,
          use_crm=False,
          use_rs=False,
          lr=5e-4,
          total_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=100,
          checkpoint_freq=10000,
          checkpoint_path=None,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          callback=None,
          load_path=None,
          eval_freq=1000,
          eval_nbr=5,
          env_name="water",
          missing=False,
          noise_level:float = 0.,
          test: bool = False,
          **network_kwargs
            ):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    network: string or a function
        neural network to use as a q function approximator. If string, has to be one of the names of registered models in baselines.common.models
        (mlp, cnn, conv_only). If a function, should take an observation tensor and return a latent variable tensor, which
        will be mapped to the Q function heads (see build_q_func in baselines.deepq.models for details on that)
    seed: int or None
        prng seed. The runs with the same seed "should" give the same results. If None, no seeding is used.
    use_crm: bool
        use counterfactual experience to train the policy
    use_rs: bool
        use reward shaping
    lr: float
        learning rate for adam optimizer
    total_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
    batch_size: int
        size of a batch sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to total_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    param_noise: bool
        whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.
    load_path: str
        path to load the model from. (default: None)
    **network_kwargs
        additional keyword arguments to pass to the network builder.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """

    if test:
        assert load_path != None, "if you want to test model, load path should be defined!"
        act = load_act(load_path)
        temp_eval_reward, temp_eval_success, _, _ = eval_model(act, eval_env, noise_level, env_name, gamma, eval_nbr = 5)
        print(temp_eval_reward)
        print(temp_eval_success)
        # python run.py --alg=deepq --env=Water-single-M3-v0 --num_timesteps=2e3 --gamma=0.9 --env_name="water" --use_crm --seed 0 --test=True --load_path="/home/mj/Projects/PartialSatLTL/psltl/baseline_algo/crm/results/water/crm/model0.pkl"
        exit()

    # Adjusting hyper-parameters by considering the number of RM states for crm
    if use_crm:
        rm_states   = env.get_num_rm_states()
        buffer_size = rm_states*buffer_size
        batch_size  = rm_states*batch_size


    # Create all the functions necessary to train the model

    sess = get_session()
    set_global_seeds(seed)

    q_func = build_q_func(network, **network_kwargs)

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph

    observation_space = env.observation_space
    def make_obs_ph(name):
        return ObservationInput(observation_space, name=name)

    act, train, update_target, debug = deepq.build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        param_noise=param_noise
    )

    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': env.action_space.n,
    }

    act = ActWrapper(act, act_params)

    # Create the replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = total_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()

    episode_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset()
    reset = True

    eval_env.reset()
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
            # Take action and update exploration to the newest value
            kwargs = {}
            if not param_noise:
                update_eps = exploration.value(t)
                update_param_noise_threshold = 0.
            else:
                update_eps = 0.
                # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                # for detailed explanation.
                update_param_noise_threshold = -np.log(1. - exploration.value(t) + exploration.value(t) / float(env.action_space.n))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = True
            action = act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
            # 0: up, 1: right, 2: down, 3: left
            if np.random.rand() < float(noise_level):
                # up or down 
                if action in [0, 2]:
                    action = np.random.choice([1, 3])
                # right or left
                elif action in [1, 3]:
                    action = np.random.choice([0, 2])

            env_action = action
            reset = False

            
            new_obs, rew, done, info = env.step(env_action)

            # Store transition in the replay buffer.
            if use_crm:
                # Adding counterfactual experience (this will alrady include shaped rewards if use_rs=True)
                experiences = info["crm-experience"]
            elif use_rs:
                # Include only the current experince but shape the reward
                experiences = [(obs, action, info["rs-reward"], new_obs, float(done))]
            else:
                # Include only the current experience (standard deepq)
                experiences = [(obs, action, rew, new_obs, float(done))]
            # Adding the experiences to the replay buffer
            for _obs, _action, _r, _new_obs, _done in experiences:
                replay_buffer.add(_obs, _action, _r, _new_obs, _done)
            
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                episode_rewards.append(0.0)
                reset = True

            if t > learning_starts and t % train_freq == 0:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if prioritized_replay:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None
                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
                if prioritized_replay:
                    new_priorities = np.abs(td_errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)

            if t > learning_starts and t % target_network_update_freq == 0:
                # Update target network periodically.
                update_target()

            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            num_episodes = len(episode_rewards)
            # if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
            #     logger.record_tabular("steps", t)
            #     logger.record_tabular("episodes", num_episodes)
            #     logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
            #     logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
            #     logger.dump_tabular()

            # if (checkpoint_freq is not None and t > learning_starts and
            #         num_episodes > 100 and t % checkpoint_freq == 0):
            #     if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
            #         if print_freq is not None:
            #             logger.log("Saving model due to mean reward increase: {} -> {}".format(
            #                        saved_mean_reward, mean_100ep_reward))
            #         save_variables(model_file)
            #         model_saved = True
            #         saved_mean_reward = mean_100ep_reward

            if t % eval_freq == 0:
                temp_eval_success, temp_eval_reward, temp_eval_partial_success, temp_eval_epi_length = eval_model(act, eval_env, noise_level, env_name, gamma, eval_nbr)
                
                total_eval_reward = sum(temp_eval_reward) / len(temp_eval_reward)
                print("step: {} and reward: {}".format(t, total_eval_reward))
                my_episode_rewards.append(temp_eval_reward)
                episode_lengths.append(temp_eval_epi_length)
                successes.append(temp_eval_success)
                partial_successes.append(temp_eval_partial_success)
                
        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            load_variables(model_file)

        if use_crm and not use_rs:
            save_path = "./results/" + env_name + "/crm"
        elif not use_crm and use_rs:
            save_path = "./results/" + env_name + "/rs"
        elif use_crm and use_rs:
            save_path = "./results/" + env_name + "/crm_rs"
        
        # act.save_act()
        
        if bool(missing):
            save_path += "_missing"

        if noise_level > 0:
            save_path += "_noise_" + str(noise_level)

        fd_path = save_path

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        act.save_act(fd_path, fd_path + "/model" + str(seed) + ".pkl")
            
        np.savez(
            save_path + "/" + str(seed),
            successes=successes,
            partial_successes=partial_successes,
            results=my_episode_rewards,
            ep_lengths=episode_lengths
                )
        
    return act
