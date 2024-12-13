# Environments

## Common
 - this folder include scafolding environments equipped with labeling map

## Skeletons
 - env_skeleton.py
    Include 
    1) measurement intialization, 
    2) setup maximum step size per episode

 - commom_ltl_env.py
    Include 
    1) automaton intialization, 
    2) setup terminal condition based on automaton
    3) setup reward functions

 - env_default_settings.py
    default setting across the environments
    1) feature representation
    2) hyperparmeter for reward shaping
    3) rolling window
    4) noise

## Ltl_envs

### cont
 - ltl_cont_env.py
    This is skeleton for continuous control environemnts
    Include
    1) step function
    2) reset function
    3) env intialization only for cont world 
    4) encode automaton state

 - ltl_water_env.py, ltl_cheetah_env.py
    Specified for each environment adjusting dimension of observation space, and action space.

### grids
 - ltl_grid_env.py
    This is skeleton for discrete environemnts
    Include
    1) step function
    2) reset function
    3) env intialization only for grid world 
    4) encode automaton state
 - ltl_office_env.py, ltl_taxi_env.py, ltl_toy_env.py
    Specified for each environment adjusting dimension of observation space, and action space.

#### maps
 - toy.txt
    This is a map for toy example. You can test with many different type of maps adding more labels to maps
    'X' is a wall that blocks agent moving into that grid map location
    'A' is an agent