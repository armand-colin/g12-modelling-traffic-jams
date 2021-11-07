from typing import List
from models.idm import IDM, IDMParameters
import sys
import numpy as np

class TestRunner:
    """Class for running tests on a IDM.
    
    The test runner is initialized with a model and a general parameters object. You then
    call the method run(), which lets you run the model multiple times, but with a varying
    value for a given parameter.
    """
    def __init__(self, model: IDM, parameters: IDMParameters):
        self.model = model
        self.parameters = parameters

    def run(self, parameter: str, value_range: List[float], jam_threshold: float, verbose=True):
        """Runs the model multiple times for each given value, then returns the time and time to jam numpy arrays.
        
        parameter: 
            name of the parameter to vary. Can be for example "a_max" for the maximum acceleration, or "v_max" for the
            maximum velocuty.
        value_range:
            range of values of which to set the parameter. The model will be run once for each value of this parameter
        jam_threshold: 
            jam threshold to estimate time to jam
        verbose:
            boolean, to print the current progress or not.
        """
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