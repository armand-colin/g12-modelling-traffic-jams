import numpy as np
from dataclasses import dataclass

@dataclass
class IDMParameters:
    v_max: float = 25.0     # maximal speed (v0)
    sx_min: float = 3.0     # minimal net distance (s0)
    T: float = 1.5          # preferred time distance
    a_max: float = 0.3      # maximum acceleration
    b: float = 3.0          # comfortable deceleration
    delta: float = 4.0      # acceleration power
    car_length: float = 5.0 # car length (l)
    N: int = 90             # number of vehicles
    L: float = 3000.0       # length of the road
    t_max: float = 100.0    # duration of the simulation
    dt: float = 0.01        # time step
    v0_noise: float = 0.2   # initial velocity noise

class IDM(IDMParameters):

    def set(self, parameters: IDMParameters):
        # Setting the parameters keys (copying the parameters into instance)
        for key in [key for key in dir(parameters) if not key.startswith('__')]:
            setattr(self, key, parameters.__dict__.get(key))

        self.one_over_2sqrtab = 1.0 / (2.0 * np.sqrt(self.a_max * self.b))
        self.iterations = int(self.t_max / self.dt)
        self.iteration = 0
        self.time = np.arange(0, self.t_max, self.dt)

        self.x  = np.zeros(shape=(self.N, self.iterations))
        self.v  = np.zeros(shape=(self.N, self.iterations))
        self.a  = np.zeros(shape=(self.N, self.iterations))
        self.dx = np.zeros(shape=(self.N, self.iterations))
        self.sx = np.zeros(shape=(self.N, self.iterations))
        self.dv = np.zeros(shape=(self.N, self.iterations))

        self.x[:, 0]  = np.arange(0, self.L, self.average_dx())[:self.N]
        self.v[:, 0]  = np.random.random(self.N) * self.v0_noise
        self.a[:, 0]  = 0.0

        self.dx[:, 0] = self.headway(self.x[:, 0])
        self.sx[:, 0] = self.dx[:, 0] - self.car_length
        self.dv[:, 0] = self.get_dv(self.v[:, 0])

        self.jam_evaluation = np.zeros(self.iterations)

    def average_dx(self):
        """Computes the average distance between vehicles, in a perfectly balanced traffic state"""
        return float(self.L) / float(self.N)

    def acceleration(self, v, dv, sx):
        """Calculates the acceleration of the model"""
        s_star = self.sx_min + v * self.T + (v * dv) * self.one_over_2sqrtab
        s_star = np.maximum(s_star, self.sx_min)

        a = self.a_max * (1.0 - np.power(v / self.v_max, self.delta) - np.power(s_star / sx, 2))
        a = np.maximum(a, -self.b)

        return a

    def headway(self, x):
        """Computes the distance of one vehicle to its leading one"""
        dx = np.zeros(self.N)
        dx[:-1] = x[1:] - x[:-1]
        dx[-1] = x[0] + self.L - x[-1]
        return dx
    
    def get_dv(self, v):
        """Computes the speed difference of a vehicle and its leading one"""
        dv = np.zeros(self.N)
        dv[:-1] = v[:-1] - v[1:]
        dv[-1] = v[-1] - v[0]
        return dv

    def integrate(self, duration: float = None):
        """Tntegrates the model. If duration is set, the model runs for the given amount of time"""
        iterations = self.iterations - 1
        
        if duration is not None:
            iterations = int(duration / self.dt)

        for _ in range(iterations):
            if self.iteration >= self.iterations - 1:
                break

            self.integration(self.iteration)
            self.iteration += 1
    
    def integration(self, i):
        """Computes the integration at the ith step"""

        self.a[:,i + 1] = self.runge_kutta_4th(self.v[:,i], self.dv[:,i], self.sx[:,i]) # + acc_noise
        self.v[:,i + 1] = np.maximum(self.v[:,i] + self.a[:,i + 1] * self.dt, 0.0)
        self.x[:,i + 1] = self.x[:,i] + self.v[:,i + 1] * self.dt

        self.dx[:,i + 1] = self.headway(self.x[:,i + 1])
        self.sx[:,i + 1] = self.dx[:,i + 1] - self.car_length
        self.dv[:,i + 1] = self.get_dv(self.v[:,i + 1])

    def euler(self, v, dv, sx):
        """Euler's integration method, used for testing"""
        return self.acceleration(v, dv, sx)

    def runge_kutta_4th(self, v, dv, sx):
        """Runge-Kutta 4th degree integration method"""
        h = self.dt

        k1 = self.acceleration(v, dv, sx)
        approx = v + k1 * h * 0.5
        approx_dv = self.get_dv(approx)

        k2 = self.acceleration(approx, approx_dv, sx)
        approx = v + k2 * h * 0.5
        approx_dv = self.get_dv(approx)

        k3 = self.acceleration(approx, approx_dv, sx)
        approx = v + k3 * h
        approx_dv = self.get_dv(approx)

        k4 = self.acceleration(approx, approx_dv, sx)

        return (k1 + 2*k2 + 2*k3 + k4) / 6.0

    def evaluate_jam(self):
        """Jam evaluation method"""
        average_distance = self.average_dx()
        average_net_distance = average_distance - self.car_length
        for i in np.arange(0, self.iterations):
            self.jam_evaluation[i] = np.sum(np.abs(self.sx[:,i] - average_net_distance))

        return self.jam_evaluation