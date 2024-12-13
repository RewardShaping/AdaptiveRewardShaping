from typing import Union
# hyperparameter for environment setup
from psltl.learner.learning_param import ContiWorldLearningParams, GridWorldLearningParams
from psltl.envs.skeletons.env_default_settings import setting, reward_kwargs
from psltl.utils.utils import set_seed
# get learners for ltl envs and rm envs
from psltl.learner.ltl_learner import ltl_env_learn, get_ltl_env


class Learner:
    def __init__(
        self, 
        params: Union[ContiWorldLearningParams, GridWorldLearningParams], 
        ):
        self.params = params
        
    def learn(
        self, 
        ) -> None:
        # params; learning params class
        params = self.params
        env_name = params.env_name
        print("Env type: ", env_name), print("=" * 75)
        
        # reward type 
        reward_type = params.reward_types
        # [True, False], [False]
        use_adrs = params.use_adrs
        use_cf = bool(params.cf)
        node_embedding = bool(params.node_embedding)
        use_one_hot = bool(params.use_one_hot)
        missing = bool(params.missing)
        # reward kwargs, this for reward shaping parameters        
        adrs_mu = 0.5

        reward_kwargs.update(dict([("hybrid_eta", params.hybrid_eta), ("adrs_update", params.adrs_update), ("adrs_mu", adrs_mu),
                                   ("reward_type", reward_type), ("adaptive_rs", use_adrs), ("theta", params.theta)
                                   ]))    
        setting.update(dict([("vector", params.vector), ("use_one_hot", params.use_one_hot), \
                                    ("adrs_update", params.adrs_update), ("node_embedding", params.node_embedding),
                                    ("missing", bool(params.missing)), ("human", params.human), ("noise", params.noise_level)
                                    ]))
        # if reward type is origin, then we use original environment without state augmentation, and LTL specification
        if reward_type == "origin":
            setting.update({"original_env": True})
        
        # we will store information about envs we will run with different cases
        algo_names = []        
        algo_name = reward_type
        
        # if we use adaptive reward shaping, then add it to the anme
        if use_adrs:
            algo_name += "_adrs"
            
        if use_cf:
            algo_name += "_cf"
        
        if node_embedding:
            algo_name += "_node_embedding"

        if use_one_hot:
            algo_name += "_one_hot"
        
        if missing:
            algo_name += "_missing"

        if params.noise_level > 0:
            algo_name += "_noise_" + str(params.noise_level)

        # save algorithm name to plot
        algo_names.append(algo_name)
        
        # fill the dictionary with the name and corresponding environment
        env, eval_env = get_ltl_env(env_name, reward_kwargs, setting, params)
        set_seed(params.seed)
        ltl_env_learn(reward_type, env, eval_env, params)