from psltl.baseline_algo.qrm.src.worlds.game import GameParams
from psltl.baseline_algo.qrm.src.worlds.taxi_world import TaxiWorldParams

class TesterTaxiWorld:
    def __init__(self, experiment, data = None):
        if data is None:
            # Reading the file
            self.experiment = experiment
            f = open(experiment)
            lines = [l.rstrip() for l in f]
            f.close()
            # setting the test attributes
            self.tasks = eval(lines[1])
        else:
            self.experiment = data["experiment"]
            self.tasks   = data["tasks"]
        # NOTE: Update the optimal value per task when you know it...
        self.optimal = {}
        for i in range(len(self.tasks)):
            self.optimal[self.tasks[i]] = 1

    def get_dictionary(self):
        d = {}
        d["experiment"] = self.experiment
        d["tasks"] = self.tasks
        d["optimal"] = self.optimal
        return d

    def get_reward_machine_files(self):
        return self.tasks

    def get_task_specifications(self):
        return self.tasks

    def get_task_params(self, task_specification):
        return GameParams("taxiworld", TaxiWorldParams())

    def get_task_rm_file(self, task_specification):
        return task_specification
