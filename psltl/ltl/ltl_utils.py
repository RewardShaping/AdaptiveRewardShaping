from itertools import product
from pythomata.utils import powerset
from psltl.ltl.partial_sat_atm import PartialSatATM
from psltl.ltl.partial_sat_atm_load import LoadedPartialSatATM
import os
import pickle


def save_atm(atm: PartialSatATM, save_info_path: str, save_delta_path: str) -> None:
    """Save automaton information with two part
	
	Parameters
	----------
	atm: PartialSatOA
		PartialSatOA class constructed by LTL specification

	save_info_path: str
		Save all the information except transition function

	save_delta_path: str
		Save transition function
	"""

    info = {}
    info["AP"] = atm.AP
    info["states"] = atm.states
    info["acc"] = atm.acc
    info["nstates"] = atm.nstates
    info["distances"] = atm.distances
    info["trapping"] = atm.trapping_q
    info["ltl"] = str(atm.ltl)

    all_subset = powerset(atm.AP)
    all_comb = product(atm.states, all_subset)

    delta = {}

    for state, transition in all_comb:
        key = (state, transition)    
        delta[key] = atm.delta(state, transition)

    with open(save_delta_path, "wb") as f:
        pickle.dump(delta, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(save_info_path, "wb") as f:
        pickle.dump(info, f, protocol=pickle.HIGHEST_PROTOCOL)
   

# get atomaton based on the environment name
def get_atm(env_name: str):
    """Get saved automaton
	
	Parameters
	----------
	env_name: str
		call automaton associated with each environment
	"""
    assert env_name in ["office", "water", "cheetah", "taxi", "toy"]
    # current file directory
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    info = dir_path + "/ltl_infos/" + env_name + "/info.pkl"
    delta = dir_path + "/ltl_infos/" + env_name + "/delta.pkl"

    atm = LoadedPartialSatATM(info, delta)

    return atm
