import pickle5 as pickle


load_path = "/home/mj/Projects/PartialSatLTL/psltl/baseline_algo/crm/results/taxi/crm/saved_model_0.pkl"
with open(load_path, 'rb') as handle:
    b = pickle.load(handle)

print(b)
