import numpy as np
import copy
from collections import defaultdict
from logaut import ltl2dfa
from pylogics.parsers import parse_ltl
from pythomata.utils import powerset


class PartialSatATM:
    """
	Attributes
	----------
	ltlquery: str
		LTL specification we want to encode

	AP: list
		Atomic proposition for the LTL

	Methods
	-------
	initialize(self) -> None:
		Initialize a distance for each automaton state before recursion

	compute_distance(self) -> dict:
		Recursively compute distance considering transition of the automaton

	get_difficulties(self, state: int) -> dict:
		Get difficulties for each state to the next states

	get_distance(self, state: int) -> float:
		Get distance for the given automaton state

	get_progression(self, state1: int, state2: int) -> float:
		Get progress from state 1 to state 2

	get_transitions(self) -> list:
		Convert all subsets of AP to match the input shape of logaut library

	delta(self, state: int, label: str) -> dict:
		Get the next automaton state for the given pari

	print_results(self, progression=False):
		Print out information for the automaton
	"""


    def __init__(self, ltlquery: str, AP: list) -> None:
        self.ltl = ltlquery
        formula = parse_ltl(ltlquery)
        self.dfa = ltl2dfa(formula, backend="lydia")
        self.AP = AP
        self.states = self.dfa._states
        self.acc = self.dfa.accepting_states
        self.trapping_q = []
        
        # Declare/initialize class variables here based on results of LTL conversion
        self.nstates = len(list(self.states)) # number of states in automaton for example
        self.max_dist = len(self.AP) * self.nstates
        self.distances = self.compute_distance()
        self.labels = 0 # save AP

    def initialize(self) -> None:
        """Go over every state of the automaton and compute distance and progression
        because you don't know the transitions you are coming from this might have to be recursive
		define distance function, first initialize, all the distance only considering the state to goal states
		"""
        
        acc = self.acc
        states = self.states
        all_transitions = self.get_transitions()
        dist = {}

        # for the acceptance states, we will give 0 distance.
        for a in acc:
            dist[a] = 0

        # states except acceptance states
        except_acc = states - acc
        # initialization run for the states except acc
        for q in except_acc:
            q_to_true = 0
            for transition in all_transitions:
                next_q = self.dfa.get_successors(q, transition)
                next_q = list(next_q)[0]
                
                if next_q in acc:
                    q_to_true += 1

            if q_to_true > 0:
                dist[q] = np.log2(2 ** len(self.AP) / q_to_true)
            else:
                dist[q] = len(self.AP) * self.nstates
                self.trapping_q.append(q)
        
        self.dist = dist

    def compute_distance(self, threshold: int = 5000) -> dict:
        """From the iteration, we will get a fixed point of the distance set
		
		Returns
		-------
		dist: dict
            Distances for each automaton states
		"""

        acc = self.acc
        states = self.states

        # first run to get distance for each state
        self.initialize()
        
        # to make a condition to stop the loop
        dist = self.dist
        prev_dist = copy.deepcopy(dist)

        # states except acceptance states
        except_acc = states - acc
        
        # loop to find fixed point of the distance set. 
        # stop condition to be safe.
        stop = 0
        while True and stop < threshold:
            updated_dist = {}
            for a in acc:
                updated_dist[a] = 0

            for q in except_acc:
                diffculties = self.get_difficulties(q)
                min_dist = dist[q]

                for nq in list(diffculties.keys()):
                    difficulty = diffculties[nq]
                    d_q_prime = dist[nq]

                    if min_dist == 0:
                        min_dist = d_q_prime + difficulty
                    elif min_dist < self.max_dist:
                        min_dist = min(min_dist, d_q_prime + difficulty)
                    else:
                        min_dist = self.max_dist

                updated_dist[q] = min_dist
            
            prev_dist = dist
            dist = updated_dist
            stop += 1
            if dist == prev_dist:
                break
        
        if stop >= threshold:
            print("the iteration terminates due to stop condition!")

        return dist  

    def get_difficulties(self, state: int) -> dict:
        """Get difficulties from the current automaton state to next automaton states
		
		Parameters
		----------
		state: int
			Automaton state

		Returns
		-------
		difficulties: dict
            Keys are next automaton state, and values are difficulty in transition from the given state to the next state
		"""

        all_transitions = self.get_transitions()
        q_to_q_prime = defaultdict(float)
        difficulties = {}

        for transition in all_transitions:
            next_q = self.dfa.get_successors(state, transition)
            next_q = list(next_q)[0]
            q_to_q_prime[next_q] += 1.

        for next_q in list(q_to_q_prime.keys()):
            difficulties[next_q] = np.log2(float(2 ** len(self.AP)) / q_to_q_prime[next_q])

        return difficulties

    def get_distance(self, state: int) -> float:
        """Get distance for the given automaton state
		
		Parameters
		----------
		state: int
            Automaton state

		Returns
		-------
		dist: float
            Distance corresponding to the given automaton state
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

    def get_transitions(self) -> list:
        """In order to match the input shape of transition derived by automaton using logaut library
		
		Returns
		-------
		all_transitions: dict
            This will return dict such as {subset of atomic proposition: True} to use logaut library
		"""

        all_subset = powerset(self.AP)
        all_transitions = []

        for subset in all_subset:
            transition = {}
            for ele in subset:
                transition[ele] = True
            all_transitions.append(transition)

        return all_transitions

    def delta(self, state: int, label: str) -> dict:
        """This will return a transmitted automaton state

        Parameters
        ----------
        state: int
            This is an automaton state

        label: str
            This is a label on the current state information

        Returns
        -------
        output: dict
            This is a transmitted automaton state for the given label and state pair e.x.) {next automaton state}
        """

        f = lambda x: dict([(l, True) for l in x])
        label = f(label)
        output = self.dfa.get_successors(state, label)

        return output

    def print_results(self, progression=False):
        """Print out information about LTL specification

        Parameters
        ----------
        progression: bool
            If this is true, it will print out what progression will be made for each transition


        """

        print("=" * 75)
        print('LTL specifiations: ', self.ltl)
        print('Atomic propositions: ', set(self.AP))
        print('Number of states: ', str(self.nstates))
        print('States: ', self.states)
        print('Acceptance set: ', self.acc)
        print('Trapping set: ', set(self.trapping_q))
        print('Distance function: ', self.distances)

        if progression:
            print('='*53, 'start to present progression between states', '='*53)

            for q in self.states:
                for nq in self.states:
                    if q != nq:
                        print('Progression form ' + str(q) + ' to ' + str(nq) + ' is ', self.get_progression(q, nq))

        print("=" * 75)


if __name__ == '__main__':
    AP = ['a', 'b', 'c', 'd']
    ltl = "(F((a & F(c))) | F((b & F(d))))"
    ltl = "F(a) & F(b) & F(c) & F(d)"
    atm = PartialSatATM(ltl, AP)
    pg0to1 = atm.get_progression(0, 1)
    atm.print_results(progression=False)
    print(atm.delta(0, 'a'))