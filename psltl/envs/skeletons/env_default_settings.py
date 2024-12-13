# default setup
reward_kwargs = dict([
    ("reward_type", "progress"), ("adaptive_rs", True), ("hybrid_eta", 0.0001), ("adrs_update", 10000),
    ("adrs_mu", 1), ("theta", "dist")
])

# this is for environment settup
setting = dict([
    ("use_one_hot", False), ("rolling", 20), ("human", False), 
    ("human_designed_reward", {}), ("noise", 0), ("violation_end", True), ("eval_nbr", 1),
    ("gamma", 0.9), ("eval_freq", 100), ("node_embedding", False),
    ("original_env", False)
])

# node_embedding: office state v1: (2, 3), state v2: [0, 0, 0, 0, 0, 1, 0 ,0, 0, 0, 0..]
# use one hot: instead of using integer value on automaton, 0 -> [1, 0, 0, 0, 0,]
# rolling: to plot, we compare only recent 'rolling (window size)'
