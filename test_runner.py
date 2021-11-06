from typing import List
from models.idm import IDM, IDMParameters
import sys
import numpy as np

class TestRunner:

    def __init__(self, model: IDM, parameters: IDMParameters):
        self.model = model
        self.parameters = parameters

    def run(self, parameter: str, value_range: List[float], jam_threshold: float, verbose=True):
        data = []

        for i in range(len(value_range)):
            
            value = value_range[i]
            
            if verbose:
                sys.stdout.write("\rProgress " + str(int(i * 1.0 / len(value_range) * 100.0)) + " % ( " + parameter + ": " + str(value) + " )")
                sys.stdout.flush()

            setattr(self.parameters, parameter, value)

            self.model.set(self.parameters)
            self.model.integrate()
            self.model.evaluate_jam()

            data.append(self.model.jam_evaluation)

        if verbose:
            print("\nDone")

        jam_times = []
        for jam_evaluation in data:
            idx = np.diff(jam_evaluation > jam_threshold, prepend=False)
            times = self.model.time[idx]
            jam_times.append(times[0] if times.size > 0 else self.parameters.t_max)
        
        return self.model.time, jam_times