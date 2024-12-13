from psltl.envs.common.grids.game_objects import Actions
import gym
import numpy as np
from collections import defaultdict


class OfficeWorld(gym.Env):
    # Containing env infos
    def __init__(self, start: tuple=(2, 1), map_size: int=1):
        self.map_size = map_size
        self.basic_map_height = 12
        self.unit_map_width = 9
        self.map_height, self.map_width = 12 * map_size, 9 * map_size
        self.feature_dim = self.map_height * self.map_width
        self.action_dim = 4
        self.start = start
        self.reset()
        self._load_map()
        
        # self.structure = self.get_structure()

    def reset(self):
        self.agent = self.start
        init_state = self.get_features()

        return init_state

    def execute_action(self, a: int) -> None:
        """
        We execute 'action' in the game
        """
        x, y = self.agent
        self.agent = self._get_new_position(x,y,a)

    def step(self, action: int):
        self.execute_action(action)

        next_state = self.get_features()
        reward = 0 # dummy reward
        done = False
        info = {}

        return next_state, reward, done, info

    def _get_new_position(self, x, y, a):
        action = Actions(a)
        # executing action
        # map is transposed with 90 degree counter clock-wise
        if (x, y, action) not in self.forbidden_transitions:
            if action == Actions.up   : y += 1
            if action == Actions.down : y -= 1
            if action == Actions.left : x -= 1
            if action == Actions.right: x += 1

        # Elevator location
        if (x, y) in self.elevator and self.map_size > 1:
            temp = self.elevator.copy()
            temp.remove((x, y))
            (x, y) = list(temp)[0]
            return x, y
        
        return x, y

    def get_events(self) -> str:
        """
        Returns the string with the propositions that are True in this state
        """
        ret = ""
        if self.agent in self.objects:
            ret += self.objects[self.agent]

        return ret

    def get_features(self) -> np.array:
        """
        Returns the features of the current state (i.e., the location of the agent)
        """
        x, y = self.agent

        return np.array([x, y])

    def get_vector_features(self) -> np.array:
        x, y = self.agent
        ret = np.zeros((self.map_height, self.map_width), dtype=np.float64)
        ret[x, y] = 1

        # this will return the position of the robot.
        return ret.ravel() # from 2D to 1D (use a.flatten() is you want to copy the array)

    def show(self):
        for y in range(self.map_width - 1, -1, -1):
            if y % 3 == 2:
                for x in range(self.map_height):
                    if x % 3 == 0:
                        print("_",end="")
                        if 0 < x < self.map_height - 1:
                            print("_",end="")
                    if (x, y, Actions.up) in self.forbidden_transitions:
                        print("_",end="")
                    else:
                        print(" ",end="")
                print()                
            for x in range(self.map_height):
                if (x, y, Actions.left) in self.forbidden_transitions:
                    print("|", end="")
                elif x % 3 == 0:
                    print(" ", end="")
                if (x, y) == self.agent:
                    print("A", end="")
                elif (x, y) in self.objects:
                    print(self.objects[(x, y)],end="")
                else:
                    print(" ",end="")
                if (x, y, Actions.right) in self.forbidden_transitions:
                    print("|",end="")
                elif x % 3 == 2:
                    print(" ",end="")
            print()      
            if y % 3 == 0:      
                for x in range(self.map_height):
                    if x % 3 == 0:
                        print("_",end="")
                        if 0 < x < self.map_height - 1:
                            print("_",end="")
                    if (x, y, Actions.down) in self.forbidden_transitions:
                        print("_",end="")
                    else:
                        print(" ",end="")
                print()                

    def get_model(self):
        """
        This method returns a model of the environment. 
        We use the model to compute optimal policies using value iteration.
        The optimal policies are used to set the average reward per step of each task to 1.
        """
        S = [(x,y) for x in range(self.map_height) for y in range(self.map_width)] # States
        A = self.actions.copy() # Actions
        L = self.objects.copy() # Labeling function
        T = {}                  # Transitions (s,a) -> s' (they are deterministic)
        for s in S:
            x, y = s
            for a in A:
                T[(s,a)] = self._get_new_position(x, y, a)

        return S, A, L, T # SALT xD

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
        self.objects[(3 * (self.map_size + 1), 3 * (self.map_size) + 1)] = "d"
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

if __name__ == "__main__":
    office = OfficeWorld(map_size=2)
    office.reset()
    office.show()
    # office._get_new_position(*(21, 1), 1)
    office.execute_action(1)
    print(office.agent)
    office.show()
    office.execute_action(2)
    office.show()
    office.execute_action(0)
    office.show()
    print(office.actions)
    # 0: up, 1: right, 2: down, 3: left