from dataclasses import dataclass
import numpy as np

@dataclass
class IDMParameters:
    v_max: float = 30.0
    dx_min: float = 2.0
    T: float = 1.5
    a_max: float = 0.73
    b: float = 1.67
    delta: float = 4.0
    car_length: float = 5.0
    N: int = 50
    L: float = 1000.0
    t_max: float = 100.0
    dt: float = 0.01

class IDM(IDMParameters):

    def set(self, parameters: IDMParameters):
        # setting the parameters keys
        for key in [key for key in dir(parameters) if not key.startswith('__')]:
            setattr(self, key, parameters.__dict__.get(key))

        self.one_over_2sq_ab = 1.0 / 2.0 * np.sqrt(self.a_max * self.b)
        self.iterations = int(self.t_max / self.dt)
        self.time = np.arange(0, self.t_max, self.dt)

        average_dx = self.steady_state_flow()

        print(average_dx)

        self.x =  np.zeros(shape=(self.N, self.iterations))
        self.v =  np.zeros(shape=(self.N, self.iterations))
        self.a =  np.zeros(shape=(self.N, self.iterations))
        self.dx = np.zeros(shape=(self.N, self.iterations))
        self.sx = np.zeros(shape=(self.N, self.iterations))
        self.dv = np.zeros(shape=(self.N, self.iterations))

        self.x[:,0] = np.arange(0, self.L, average_dx)[:self.N]
        self.v[:,0] = 1.0
        self.a[:,0] = 0.0

        self.dx[:,0] = self.headway(self.x[:,0])
        self.sx[:,0] = self.dx[:,0] - self.car_length
        self.dv[:,0] = self.get_dv(self.v[:,0])

        print(self.x[:,0])

    def steady_state_flow(self):
        average_dx = float(self.L) / float(self.N)
        return average_dx

    def acceleration(self, v, dv, sx):
        s_star = self.dx_min + v * self.T + v * dv * self.one_over_2sq_ab
        a = self.a_max * (1.0 - np.power(v / self.v_max, self.delta) - np.power(s_star / sx, 2))
        return a

    def headway(self, x):
        dx = np.zeros(self.N)
        dx[:-1] = x[1:] - x[:-1]
        dx[-1] = x[0] + self.L - x[-1]
        return dx
    
    def get_dv(self, v):
        dv = np.zeros(self.N)
        dv[:-1] =  v[1:] - v[:-1]
        dv[-1] = v[0] - v[-1]
        return dv

    def integrate(self):
        for i in range(0, self.iterations - 1):
            self.integration(i)
    
    """Computes the integration at the ith step"""
    def integration(self, i):
        self.a[:,i + 1] = self.runge_kutta_4th(self.v[:,i], self.dv[:,i], self.sx[:,i])
        self.v[:,i + 1] = self.v[:,i] + self.a[:,i + 1] * self.dt
        self.x[:,i + 1] = self.x[:,i] + self.v[:,i + 1] * self.dt

        self.dx[:,i + 1] = self.headway(self.x[:,i + 1])
        self.sx[:,i + 1] = self.dx[:,i + 1] - self.car_length
        self.dv[:,i + 1] = self.get_dv(self.v[:,i + 1])

    """Runge-Kutta 4th degree integration method"""
    def runge_kutta_4th(self, v, dv, sx):
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

    def runge_kutta_3rd(self, v, dv, sx):
        h = self.dt

        k1 = self.acceleration(v, dv, sx)
        approx = v + k1 * h * 0.5