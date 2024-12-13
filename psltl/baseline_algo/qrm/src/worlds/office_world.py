if __name__ == '__main__':
    # This is a terrible hack just to be able to execute this file directly
    import sys
    sys.path.insert(0, '../')

from psltl.baseline_algo.qrm.src.worlds.game_objects import Actions
import random, math, os
import numpy as np
from collections import defaultdict
"""
Auxiliary class with the configuration parameters that the Game class needs
"""


class OfficeWorldParams:
    def __init__(self, start=(2, 1), map_size=1):
        self.start = start
        self.map_size = map_size

class OfficeWorld:

    def __init__(self, params):
        self.env_game_over = False
        self.start = params.start
        self.map_size = params.map_size
        map_size = params.map_size
        self.agent = params.start
        self.basic_map_height = 12
        self.unit_map_width = 9
        self.map_height = 12 * map_size
        self.map_width = 9 * map_size
        self.feature_dim = self.map_height * self.map_width
        self.action_dim = 4
        self._load_map()        

    def execute_action(self, a):
        """
        We execute 'action' in the game
        """
        action = Actions(a)
        x,y = self.agent
        # executing action
        if (x,y,action) not in self.forbidden_transitions:
            if action == Actions.up   : y+=1
            if action == Actions.down : y-=1
            if action == Actions.left : x-=1
            if action == Actions.right: x+=1
        self.agent = (x,y)

    def get_actions(self):
        """
        Returns the list with the actions that the agent can perform
        """
        return self.actions

    def get_true_propositions(self):
        """
        Returns the string with the propositions that are True in this state
        """
        ret = ""
        if self.agent in self.objects:
            ret += self.objects[self.agent]
        return ret

    def get_state(self):
        return None # we are only using "simple reward machines" for the craft domain

    # The following methods return different feature representations of the map ------------
    def get_features(self):
        x,y = self.agent
        ret = np.zeros((self.map_height, self.map_width), dtype=np.float64)
        ret[x, y] = 1

        # this will return the position of the robot.
        return ret.ravel() # from 2D to 1D (use a.flatten() is you want to copy the array)

    def show(self):
        for y in range(8,-1,-1):
            if y % 3 == 2:
                for x in range(12):
                    if x % 3 == 0:
                        print("_",end="")
                        if 0 < x < 11:
                            print("_",end="")
                    if (x,y,Actions.up) in self.forbidden_transitions:
                        print("_",end="")
                    else:
                        print(" ",end="")
                print()                
            for x in range(12):
                if (x,y,Actions.left) in self.forbidden_transitions:
                    print("|",end="")
                elif x % 3 == 0:
                    print(" ",end="")
                if (x,y) == self.agent:
                    print("A",end="")
                elif (x,y) in self.objects:
                    print(self.objects[(x,y)],end="")
                else:
                    print(" ",end="")
                if (x,y,Actions.right) in self.forbidden_transitions:
                    print("|",end="")
                elif x % 3 == 2:
                    print(" ",end="")
            print()      
            if y % 3 == 0:      
                for x in range(12):
                    if x % 3 == 0:
                        print("_",end="")
                        if 0 < x < 11:
                            print("_",end="")
                    if (x,y,Actions.down) in self.forbidden_transitions:
                        print("_",end="")
                    else:
                        print(" ",end="")
                print()                

    # The following methods create the map ----------------------------------------------
    def _load_map(self):
        # Creating the map
        self.objects = defaultdict(lambda: "")
        # version original
        self.objects[(1, 1)] = "a"
        self.objects[(1, self.map_width - 2)] = "d"
        self.objects[(self.map_height - 2, self.map_width - 2)] = "c"

        self.elevator = []
        if self.map_size == 1:
            self.objects[(self.map_height - 2, 1)] = "b"
        else:
            for i in range(self.map_size):
                self.objects[(self.map_height - 2, 1 + i * self.unit_map_width)] = "l"
                self.elevator.append((self.map_height - 2, 1 + i * self.unit_map_width))

        self.objects[(3 * (self.map_size + 1) + 1, 3 * (self.map_size) + 1)] = "e" # MAIL
        self.objects[(3 * (self.map_size + 1) + 2, 3 * (self.map_size - 1) + 2)] = "f" # COFFEE
        self.objects[(3 * (self.map_size), 3 * (self.map_size + 1))] = "f" # COFFEE
        self.objects[(3 * (self.map_size) + 1, 3 * (self.map_size + 2*(self.map_size - 1)) + 1)] = "g"  # OFFICE

        for i in range(self.map_height // 3):
            for j in range(self.map_height // 3):
                if len(self.objects[(1 + 3 * i, 1 + 3 * j)]) == 0:
                    self.objects[(1 + 3 * i, 1 + 3 * j)] = "n"

        # Adding walls
        self.forbidden_transitions = set()

        # general grid
        for x in range(self.map_height):
            for y in np.arange(0, self.map_width, 3):
                self.forbidden_transitions.add((x, y, Actions.down)) 
                self.forbidden_transitions.add((x, y + 2, Actions.up))

        for y in range(self.map_width):
            for x in np.arange(0, self.map_height, 3):
                self.forbidden_transitions.add((x, y, Actions.left))
                self.forbidden_transitions.add((x + 2, y, Actions.right))
        door_idxex = [1 + i * self.unit_map_width for i in range(self.map_size)] + [7 + i * 9 for i in range(self.map_size)]
        # adding 'doors'
        for y in door_idxex:
            for x in (np.arange(1, self.map_height - 1, 3) + 1):
                if x >= self.map_height - 1:
                    # beyond map
                    pass
                else:
                    self.forbidden_transitions.remove((x, y, Actions.right))
                    self.forbidden_transitions.remove((x + 1, y, Actions.left))

        for x in np.arange(1, self.map_height, 3):
            for i in range(self.map_size):
                if 5 + i * self.unit_map_width >= self.map_width - 1:
                    # beyond map
                    pass
                else:
                    self.forbidden_transitions.remove((x, 5 + i * self.unit_map_width, Actions.up))
                    self.forbidden_transitions.remove((x, 6 + i * self.unit_map_width, Actions.down))

        for x in np.arange(1, self.map_height, self.map_width):
            for i in range(self.map_size):
                if 5 + i * self.unit_map_width >= self.map_width - 1:
                    # beyond map
                    pass
                else:
                    self.forbidden_transitions.remove((x, 2 + i * self.unit_map_width, Actions.up))
                    self.forbidden_transitions.remove((x, 3 + i * self.unit_map_width, Actions.down))
                
        # Adding the agent
        self.actions = [Actions.up.value, Actions.right.value, Actions.down.value, Actions.left.value]

def play():
    from reward_machines.reward_machine import RewardMachine

    # commands
    str_to_action = {"w":Actions.up.value,"d":Actions.right.value,"s":Actions.down.value,"a":Actions.left.value}
    params = OfficeWorldParams()

    # play the game!
    tasks = ["../../experiments/office/reward_machines/t%d.txt"%i for i in [1,2,3,4]]
    reward_machines = []
    for t in tasks:
        reward_machines.append(RewardMachine(t))
    for i in range(len(tasks)):
        print("Running", tasks[i])

        game = OfficeWorld(params) # setting the environment
        rm = reward_machines[i]  # setting the reward machine
        s1 = game.get_state()
        u1 = rm.get_initial_state()
        while True:
            # Showing game
            game.show()
            print("Events:", game.get_true_propositions())
            #print(game.getLTLGoal())
            # Getting action
            print("u:", u1)
            print("\nAction? ", end="")
            a = input()
            print()
            # Executing action
            if a in str_to_action:
                game.execute_action(str_to_action[a])

                # Getting new state and truth valuation
                s2 = game.get_state()
                events = game.get_true_propositions()
                u2 = rm.get_next_state(u1, events)
                r = rm.get_reward(u1,u2,s1,a,s2)
                
                # Getting rewards and next states for each reward machine
                rewards, next_states = [],[]
                for j in range(len(reward_machines)):
                    j_rewards, j_next_states = reward_machines[j].get_rewards_and_next_states(s1, a, s2, events)
                    rewards.append(j_rewards)
                    next_states.append(j_next_states)
                
                print("---------------------")
                print("Rewards:", rewards)
                print("Next States:", next_states)
                print("Reward:", r)
                print("---------------------")
                
                if game.env_game_over or rm.is_terminal_state(u2): # Game Over
                    break 
                
                s1 = s2
                u1 = u2
            else:
                print("Forbidden action")
        game.show()
        print("Events:", game.get_true_propositions())
    
# This code allow to play a game (for debugging purposes)
if __name__ == '__main__':
    play()
