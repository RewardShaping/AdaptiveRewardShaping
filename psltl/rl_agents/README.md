# rl_agents
 

## Common
 - callback.py is customized in order to track of success rate, and automaton states
 - evaluation.py is customized in order to apply a discount factor to compute expected return
 - base_class.py, off_policy_algorithm.py are customized in order to implement DDQN which is not originally implemented in stable baselines 3
 - scheduler.py, customized scheduler for learning rate and exploration rate
 - torch_layer.py, is customized in order to switch on and off for bias term

## DQN
 - DDQN option is added
 - When we use linear-Neural Network, we enable the algorithm turn on and off for bias term
