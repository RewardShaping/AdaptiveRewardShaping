import numpy as np
from typing import List, Union
from collections import defaultdict
import numpy as np
import pickle


class LoadedPartialSatATM:
    """
    Attributes
    ----------
    info_path: str
        path for saved information of automaton, which contain how many automaton states, what is trapping state, what is acceptance states extra

    delta_path: str
        path for saved information of automaton, especially for transition map for the automaton

    Methods
    -------
    get_rank(self) -> Dict[list]:
        This method gets rank for automtaton based on distance

    get_distance(self, state: int) -> float:
        This method will return a distance value for each automaton state

    get_progression(self, state1: int, state2: int) -> float:
        This will reutrn progression when automaton state is transmitted

    get_state_matching(self, state_matching: np.array) -> None:
        We might use a different value for each automaton instead of using integer value to distinguish different states

    get_sorted_states(self) -> dict:
        This will return sorted dictionary according to values of the dictionary

    encode_label(self, label: List[Union(str, tuple)]) -> frozenset:
        We call loaded information which is string, and we will convert string to map using 'eval' fucnction

    delta(self, state: int, label: Union(List[Union(str, tuple)], str)) -> dict:
        Get transition of the current automaton state with the given label

    print_results(self, progression=False, delta=False) -> None:
        Print information of automaton
    """
    
    def __init__(self, info_path: str, delta_path: str) -> None:
        self.dict_delta = {}
        with open(delta_path, "rb") as f:
            self.dict_delta = pickle.load(f)

        self.info = {}
        with open(info_path, "rb") as f:
            self.info = pickle.load(f)
        
        self.ltl = self.info["ltl"]
        self.AP = self.info["AP"]
        self.states = self.info["states"]
        self.acc = self.info["acc"]
        self.nstates = self.info["nstates"]
        self.distances = self.info["distances"]
        self.trapping_q = self.info["trapping"]
        self.non_terminal_states = self.states - set(self.acc) - set(self.trapping_q)

        # sorted states and distance
        # [(state, distance)]
        self.sorted_states_and_distance = self.get_sorted_states()
        self.rank = self.get_rank()
        self.total_rank = len(self.rank.keys())
        self.state2next_states = defaultdict(set)
        for x, y in self.dict_delta.items():
            _in = x[0]
            _out = list(y)[0]
            self.state2next_states[_in].add(list(y)[0])
                
    def get_next_states(self, state):
        return self.state2next_states[state]

    def get_rank(self) -> dict:
        """We design ranked structure based on distance
		
		Returns
		-------
		rank: dict
            Keys are distance, and values are automaton state. So we get the rank based on distance for a given automaton state.
            Distance will be float, and automaton states will be integer valued
		"""

        rank = defaultdict(list)
        
        for s, d in self.sorted_states_and_distance:
            rank[float(d)].append(int(s)) 

        return rank

    def get_distance(self, state: int) -> float:
        """Get distance for the given automaton state
		
		Parameters
		----------
		state: int
			Automaton state

		Returns
		-------
		dist: float
            Distance of the given automaton state
		"""        

        dist = self.distances[state]

        return dist

    def get_progression(self, state1: int, state2: int) -> float:
        """Evaluate progression when transition from state 1 to state 2 is made
		
		Parameters
		----------
		state1: int
			Previous automaton state

		state2: int
			Next automaton state

		Returns
		-------
		progress: float
            this will return progress made by the transition, which will be always bigger or equal than 0
		"""

        progress = max(0, self.distances[state1] - self.distances[state2])

        return progress

    def get_state_matching(self, state_matching: np.array) -> None:
        """We may want to change the value of automaton state
		
		Parameters
		----------
		state_matching: np.array
			This is a map corresponding each automaton state to other values e.x.) int -> array

		"""

        self.state_matching = state_matching

    def get_sorted_states(self) -> dict:
        """Get sorted dictionary whose keys are states and values are distances
		
		Returns
		-------
		sorted_states_and_distance: dict
            We get sorted dictionary by distance for each automaton state
		"""

        sorted_states_and_distance = sorted(self.distances.items(), key=lambda x:x[1])

        return sorted_states_and_distance

    def encode_label(self, label: str) -> frozenset:
        """This function will be used to convert 'string' to 'frozenset' (Note that we get transition map from saved string type)

        Parameters
        ----------
        label: List[Union(str, tuple)]
            This is list of labels whose value is string or string inside tuple e.x.) (a, )

        Returns
        -------
        change_from_str: frozenset
            Encode string type labels to frozenset in order to use transition map
        """
        assert type(label) == str, "type of label is not string! instead, the type is {}".format(type(label))
        not_in_AP = False
        for l in label:
            if l not in self.AP:
                not_in_AP = True
                break
        
        if not_in_AP or label == "" or label == " ":
            wrapped_string_label = 'frozenset()'
        else:
            string_label = ""

            for l in label[:-1]:
                string_label += "'" + l + "', "

            string_label += "'" + label[-1] + "'"
            wrapping_function = lambda x: "frozenset({" + str(x) +"})"
            wrapped_string_label = wrapping_function(string_label)

        change_from_str = eval(wrapped_string_label)

        return change_from_str

    def delta(self, state: int, label: str) -> dict:
        """This will return transmitted automaton state for the given label and state pair

        Parameters
        ----------
        state: int
            This is an automaton state

        label: str
            This is a label on the current state information

        Returns
        -------
        output: dict
            This is a transmitted automaton state for the given label and state pair
        """
        
        wrapped_string_label = self.encode_label(label)
        output = self.dict_delta[(state, wrapped_string_label)]

        return output

    def print_results(self, progression: bool=False, delta: bool=False) -> None:
        """Print out information about LTL specification

        Parameters
        ----------
        progression: bool
            If this is true, it will print out what progression will be made for each transition

        delta: bool
            If this is true, it will print out transition map

        """

        print("=" * 75)
        print('LTL specifiations: ', self.ltl)
        print('Atomic propositions: ', set(self.AP))
        print('Number of states: ', str(self.nstates))
        print('States: ', self.states)
        print('Acceptance set: ', self.acc)
        print('Trapping set: ', set(self.trapping_q))
        print('Distance function: ', self.distances)
        if delta:
            print("Delta", self.dict_delta)

        if progression:
            print('='*53, 'start to present progression between states', '='*53)

            for q in self.states:
                for nq in self.states:
                    if q != nq:
                        print('Progression form ' + str(q) + ' to ' + str(nq) + ' is ', self.get_progression(q, nq))

        print("=" * 75)

if __name__ == '__main__':
    basic_path = "./ltl_infos"
    info_path = basic_path + "/office/info.pkl"
    delta_path = basic_path + "/office/delta.pkl"
    atm = LoadedPartialSatATM(info_path, delta_path)
    pg0to1 = atm.get_progression(0, 1)
    atm.print_results(progression=False)
    print(atm.dict_delta.keys())
    print(atm.states)
    print(atm.AP)
    print(atm.trapping_q)
    print(atm.state2next_states)
    print(atm.get_next_states(0))
    print(atm.delta(0, 'fe'))