if __name__ == '__main__':
    # This is a terrible hack just to be able to execute this file directly
    import sys
    sys.path.insert(0, '../')

from psltl.baseline_algo.crm.envs.grids.game_objects import Actions
import random, math, os
import numpy as np

import gymnasium as gym
import numpy as np

# render_mode will be 'human', 'None'

class TaxiWorldParams:
    def __init__(self, max_count:int = 20, seed: int = 0):
        self.max_count = max_count
        self.seed = seed


class Taxi(gym.Wrapper):
    def __init__(self, max_count: int=20, seed=0):
        env = gym.make('Taxi-v3', render_mode=None)
        super().__init__(env)
        self.seed = seed
        self.env = env
        self.curr_mdp_state, _ = env.reset()
        self.curr_label = ""
        self.max_count = max_count
        self.map_height = 1
        self.map_width = 500
        self.episode_step = 0

        # actions [0, 1, 2, 3, 4, 5]
        self.pickup_count = 0 # action is 4
        self.dropoff_count = 0 # action is 5
        self.actions = [i for i in range(self.action_space.n)]
        self.reset()

    def get_events(self):

        label = self.curr_label 
        return label 
    
    def get_actions(self):
        
        return self.actions
    
    def get_count(self):

        total_count = self.pickup_count + self.dropoff_count
        return total_count

    def get_location(self, idx):
        
        return self.env.locs[idx]
    
    def get_features(self):

        return self.curr_mdp_state
    
    def get_vector_features(self):
        
        return np.identity(self.map_height * self.map_width)[self.curr_mdp_state]
    
    def get_true_propositions(self):

        return self.curr_label
    
    def get_state(self):
        return None # we are only using "simple reward machines" for the craft domain

    def execute_action(self, action):
        self.step(action)

    def step(self, action):
        self.episode_step += 1
        
        next_state, reward, _, done, info = self.env.step(action)        
        self.curr_mdp_state = next_state
        taxi_row, taxi_col, pass_loc, dest_idx = self.env.decode(self.curr_mdp_state)
        self.curr_label = ""
        
        # reach destination regardless of passengers
        if ((taxi_row, taxi_col) == self.get_location(dest_idx)):
            self.curr_label += "l"
        
        if pass_loc == 4:
            self.curr_label += "p"
            self.have_passenger = 1
        
        # while the taxi has passenger
        if self.have_passenger:
            self.curr_label += "p"
            if action == 5:
                self.curr_label += "d"
                self.have_passenger = 0
                # drop-off to the destinaion
                if "l" in self.curr_label:
                    self.curr_label += "g"

        else:
            if action == 5:
                self.curr_label += "d"
            # pick up passenger
            elif action == 4 and ((taxi_row, taxi_col) == self.get_location(pass_loc)):
                self.curr_label += "p" # get passenger
                self.have_passenger = 1
        
        self.env_game_over = done
        return next_state, reward, done, info
    
    def reset(self):
        init_state = self.env.reset()
        self.curr_mdp_state, _ = init_state
        self.curr_label = ""
        self.episode_step = 0
        self.pickup_count = 0 
        self.dropoff_count = 0
        self.have_passenger = 0
        self.done = False

        return self.curr_mdp_state