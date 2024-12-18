from psltl.baseline_algo.qrm.src.tester.tester_craft import TesterCraftWorld
from psltl.baseline_algo.qrm.src.tester.tester_office import TesterOfficeWorld
from psltl.baseline_algo.qrm.src.tester.tester_water import TesterWaterWorld
from psltl.baseline_algo.qrm.src.tester.tester_taxi import TesterTaxiWorld
# from reward_machines.reward_machine import RewardMachine
from psltl.baseline_algo.qrm.src.reward_machines.reward_machine_mine import RewardMachine
from psltl.baseline_algo.qrm.src.tester.test_utils import read_json, get_precentiles_str, get_precentiles_in_seconds, reward2steps
import pickle5 as pickle
import numpy as np
import time, os

class Tester:
    def __init__(self, learning_params, testing_params, experiment, use_rs, result_file=None):
        if result_file is None: # in this case, we are running a new experiment
            self.learning_params = learning_params
            self.testing_params = testing_params
            # Reading the file
            self.experiment = experiment
            f = open(experiment)
            lines = [l.rstrip() for l in f]
            f.close()

            # setting the right world environment
            self.game_type = eval(lines[0])
            if self.game_type == "officeworld":
                self.world = TesterOfficeWorld(experiment, learning_params.gamma)
            if self.game_type == "craftworld":
                self.world = TesterCraftWorld(experiment, learning_params.tabular_case, learning_params.gamma)
            if self.game_type == "waterworld":
                self.world = TesterWaterWorld(experiment, learning_params.use_random_maps)
            elif self.game_type == "taxiworld":
                self.world = TesterTaxiWorld(experiment)

            # Creating the reward machines for each task
            self.reward_machines = []
            self.file_to_reward_machine = {}
            rm_files = self.world.get_reward_machine_files()
            for i in range(len(rm_files)):
                rm_file = rm_files[i]
                self.file_to_reward_machine[rm_file] = i
                self.reward_machines.append(RewardMachine(rm_file, use_rs, learning_params.gamma))

            # I store the results here
            self.results = {}
            self.steps = []
            self.results["eval_rewards"] = []
            self.results["mdp_rewards"] = []
            self.results["rm_states"] = []
            self.results["last_rm_states"] = [] 
            self.results["ep_lengths"] = [] 
            aux_tasks = self.get_task_specifications()
            for i in range(len(aux_tasks)):
                t_str = str(aux_tasks[i])
                self.results[t_str] = {}

        else:
            # In this case, we load the results that were precomputed in a previous run
            data = read_json(result_file)
            self.game_type = data['game_type']
            if self.game_type == "craftworld":
                self.world = TesterCraftWorld(None, None, None, data['world'])
            if self.game_type == "waterworld":
                self.world = TesterWaterWorld(None, None, data['world'])
            if self.game_type == "officeworld":
                self.world = TesterOfficeWorld(None, None, data['world'])
            elif self.game_type == "taxiworld":
                self.world = TesterTaxiWorld(None, data["world"])

            self.results = data['results']
            self.steps   = data['steps']            
            # obs: json transform the interger keys from 'results' into strings
            # so I'm changing the 'steps' to strings
            for i in range(len(self.steps)):
                self.steps[i] = str(self.steps[i])

    def get_world_name(self):
        return self.game_type.replace("world","")

    def get_task_params(self, task_specification):
        return self.world.get_task_params(task_specification)

    def get_reward_machine_id_from_file(self, rm_file):
        return self.file_to_reward_machine[rm_file]

    def get_reward_machine_id(self, task_specification):
        rm_file = self.world.get_task_rm_file(task_specification)
        return self.get_reward_machine_id_from_file(rm_file)

    def get_reward_machines(self):
        return self.reward_machines

    def get_world_dictionary(self):
        return self.world.get_dictionary()
    
    def get_task_specifications(self):
        # Returns the list with the task specifications (reward machine + env params)
        return self.world.get_task_specifications()

    def get_task_rms(self):
        # Returns only the reward machines that we are learning
        return self.world.get_reward_machine_files()

    def get_optimal(self, task_specification):
        r = self.world.optimal[task_specification]
        return r if r > 0 else 1.0
    
    def save_results(self, path):
        task_specification = self.get_task_specifications()[0]
        if not os.path.exists(path):
            os.makedirs(path)

        save_path = path + "/" + str(task_specification) + ".pkl"
        with open(save_path, 'wb') as handle:
            pickle.dump(self.results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def run_test(self, step, sess, test_function, *test_args):
        t_init = time.time()
        # 'test_function' parameters should be (sess, task_params, learning_params, testing_params, *test_args)
        # and returns the reward
        reward_machines = self.get_reward_machines()
        aux = []
        for task_specification in self.get_task_specifications():
            task_str = str(task_specification)
            task_params = self.get_task_params(task_specification)
            task_rm_id  = self.get_reward_machine_id(task_specification)
            eval_reward, mdp_reward, rm_state, last_rm_state, epi_length = test_function(sess, reward_machines, task_params, task_rm_id, self.learning_params, self.testing_params, *test_args)
            reward = mdp_reward
            if step not in self.results[task_str]:
                self.results[task_str][step] = []
            if len(self.steps) == 0 or self.steps[-1] < step:
                self.steps.append(step)
            if reward is None:
                # the test returns 'none' when, for some reason, this network hasn't change
                # so we have to copy the results from the previous iteration
                id_step = [i for i in range(len(self.steps)) if self.steps[i] == step][0] - 1
                reward = 0 if id_step < 0 else self.results[task_str][self.steps[id_step]][-1]
                #print("Skiped reward is", reward)
            # self.results[task_str][step].append(reward)
            self.results["eval_rewards"].append(eval_reward)
            self.results["mdp_rewards"].append(mdp_reward)
            self.results["rm_states"].append(rm_state)
            self.results["last_rm_states"].append(last_rm_state)
            self.results["ep_lengths"].append(epi_length)
            aux.append(reward)

        print("Testing: %0.1f"%(time.time() - t_init), "seconds\tTotal: %d"%sum([(r if r > 0 else self.testing_params.num_steps) for r in reward2steps(aux)]))
        print("\t".join(["%d"%(r) for r in reward2steps(aux)]))
        print("ep_lengths", epi_length)
        print("eval r total", eval_reward)
        print("mdp reward", mdp_reward)
        print("rm state just before the last:", rm_state)
        print("last rm state:", last_rm_state)
        # print("mdp r total", mdp_reward)
        

    def show_results(self):
        average_reward = {}
        
        tasks = self.get_task_specifications()

        # Showing perfomance per task
        for t in tasks:
            t_str = str(t)
            print("\n" + t_str + " --------------------")
            print("steps\tP25\t\tP50\t\tP75")            
            for s in self.steps:
                normalized_rewards = [r/self.get_optimal(t) for r in self.results[t_str][s]]
                a = np.array(normalized_rewards)
                if s not in average_reward: average_reward[s] = a
                else: average_reward[s] = a + average_reward[s]
                p25, p50, p75 = get_precentiles_str(a)
                #p25, p50, p75 = get_precentiles_in_seconds(a)
                print(str(s) + "\t" + p25 + "\t" + p50 + "\t" + p75)

        # Showing average perfomance across all the task
        print("\nAverage Reward --------------------")
        print("steps\tP25\t\tP50\t\tP75")            
        num_tasks = float(len(tasks))
        for s in self.steps:
            normalized_rewards = average_reward[s] / num_tasks
            p25, p50, p75 = get_precentiles_str(normalized_rewards)
            #p25, p50, p75 = get_precentiles_in_seconds(normalized_rewards)
            print(str(s) + "\t" + p25 + "\t" + p50 + "\t" + p75)

    def get_best_performance_per_task(self):
        # returns the best performance per task (this is relevant for reward normalization)
        ret = {}
        for t in self.get_task_specifications():
            t_str = str(t)
            ret[t_str] = max([max(self.results[t_str][s]) for s in self.steps]) 
        return ret

    def get_result_summary(self):
        """
        Returns normalized average performance across all the tasks
        """
        average_reward = {}
        task_reward = {}
        task_reward_count = {}
        tasks = self.get_task_specifications()

        # Computing average reward per task
        for task_specification in tasks:
            t_str = str(task_specification)
            task_rm = self.world.get_task_rm_file(task_specification)
            if task_rm not in task_reward:
                task_reward[task_rm] = {}
                task_reward_count[task_rm] = 0
            task_reward_count[task_rm] += 1
            for s in self.steps:
                normalized_rewards = [r/self.get_optimal(t_str) for r in self.results[t_str][s]]
                a = np.array(normalized_rewards)
                # adding to the average reward
                if s not in average_reward: average_reward[s] = a
                else: average_reward[s] = a + average_reward[s]
                # adding to the average reward per tas
                if s not in task_reward[task_rm]: task_reward[task_rm][s] = a
                else: task_reward[task_rm][s] = a + task_reward[task_rm][s]

        # Computing average reward across all tasks
        ret = []
        ret_task = {}
        for task_rm in task_reward:
            ret_task[task_rm] = []
        num_tasks = float(len(tasks))
        for s in self.steps:
            normalized_rewards = average_reward[s] / num_tasks
            ret.append([s, normalized_rewards])
            for task_rm in task_reward:
                normalized_task_rewards = task_reward[task_rm][s] / float(task_reward_count[task_rm])
                ret_task[task_rm].append([s, normalized_task_rewards])
        ret_task["all"] = ret
                
        return ret_task

