import numpy as np
from dataclasses import dataclass

@dataclass
class OVMParameters():
    N: int = 100
    L: int = 200
    s: float = 1.0
    t_max: float = 1000.0
    dt: float = 0.1
    velocity_function: str = "tanh"
    max_speed: float = 2.0
    max_distance: float = 4.0
    r: float = 10.0

class OVM(OVMParameters):

    def set(self, parameters: OVMParameters):
        # setting the parameters keys
        for key in [key for key in dir(parameters) if not key.startswith('__')]:
            setattr(self, key, parameters.__dict__.get(key))

        self.iterations = int(self.t_max / self.dt)
        self.time = np.arange(0, self.t_max, self.dt)

        # defining velocity function
        if self.velocity_function == "logistic":
            self.V = self.V_logistic
        elif self.velocity_function == "linear": 
            self.V = self.V_linear
        else: # default 
            self.V = self.V_tanh

        average_dx, average_speed = self.steady_state_flow()

        self.x =  np.zeros(shape=(self.N, self.iterations))
        self.v =  np.zeros(shape=(self.N, self.iterations))
        self.a =  np.zeros(shape=(self.N, self.iterations))
        self.dx = np.zeros(shape=(self.N, self.iterations))

        self.x[:,0] = np.arange(0, self.L, average_dx)[:self.N]
        self.v[:,0] = average_speed
        self.a[:,0] = 0.0

        self.dx[:,0] = self.headway(self.x[:,0])

        self.jam_evaluation = np.zeros(self.iterations)


    def V(self, x):
        pass

    def V_tanh(self, x):
        return (np.tanh((x * 4.0 / self.max_distance) - 2) + np.tanh(2)) * 0.5 * self.max_speed
    
    def V_linear(self, x):
        return np.clip(x * self.max_speed / self.max_distance, 0.0, self.max_speed)
    
    def V_logistic(self, x):
        return self.max_speed / (1.0 + np.exp(-self.r * (x / self.max_distance - 0.5)))


    """Returns the average distance between vehicles and the speed at this average distance"""
    def steady_state_flow(self):
        average_dx = float(self.L) / float(self.N)
        average_speed = self.V(average_dx)
        return average_dx, average_speed

    """Returns the dx array corresponding to the given x array"""
    def headway(self, x: np.array):
        dx = np.zeros(self.N)
        dx[:-1] = x[1:] - x[:-1]
        dx[-1] = x[0] + self.L - x[-1]
        return dx

    """Acceleration definiton of the OVM"""
    def acceleration(self, dx, v):
        return self.s * (self.V(dx) - v)

    """Computes the integration of the model"""
    def integrate(self):
        for i in range(0, self.iterations - 1):
            self.integration(i)
    
    """Computes the integration at the ith step"""
    def integration(self, i):
        self.a[:,i + 1] = self.runge_kutta_4th(self.dx[:,i], self.v[:,i])
        self.v[:,i + 1] = self.v[:,i] + self.a[:,i + 1] * self.dt
        self.x[:,i + 1] = self.x[:,i] + self.v[:, i + 1] * self.dt

        self.dx[:,i + 1] = self.headway(self.x[:, i + 1])


    """Runge-Kutta 4th degree integration method"""
    def runge_kutta_4th(self, dx, v):
        h = self.dt

        k1 = self.acceleration(dx, v)
        approx = v + k1 * h * 0.5
        k2 = self.acceleration(dx, approx)
        approx = v + k2 * h * 0.5
        k3 = self.acceleration(dx, approx)
        approx = v + k3 * h
        k4 = self.acceleration(dx, approx)

        return (k1 + 2*k2 + 2*k3 + k4) / 6.0

    def evaluate_jam(self):
        average_distance, _ = self.steady_state_flow()
        for i in np.arange(0, self.iterations):
            self.jam_evaluation[i] = np.sum(np.abs(self.dx[:,i] - average_distance))

        return self.jam_evaluation