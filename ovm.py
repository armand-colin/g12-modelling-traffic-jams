import numpy as np
import math
from enum import Enum, auto

class ovm:

    def __init__(self):
        pass
    
    def set(
        self,
        N: int = 100,
        L: int = 200,
        sensitivity: float = 1.0,
        tmax: float = 1000.0,
        dt: float = 0.1,
        velocity_function: str = "tanh",
        max_speed: float = 2.0
    ):
        self.N = N
        self.L = L
        self.dt = dt
        self.tmax = tmax
        self.sensitivity = sensitivity

        self.iterations = int(tmax / dt)
        self.time = np.arange(0, self.tmax, self.dt)
        self.max_speed = max_speed

        # defining velocity function
        if velocity_function == "tanh":
            self.V = self.V_tanh
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

    def V(self, x):
        pass

    def V_tanh(self, x):
        return (np.tanh(x - 2) + np.tanh(2)) * 0.5 * self.max_speed

    def steady_state_flow(self):
        average_dx = float(self.L) / float(self.N)
        average_speed = self.V(average_dx)
        return average_dx, average_speed

    def headway(self, x):
        dx = np.zeros(self.N)
        dx[:-1] = x[1:] - x[:-1]
        dx[-1] = x[0] + self.L - x[-1]
        return dx

    def integrate(self):
        for i in range(0, self.iterations - 1):
            self.integration(i)
    
    def integration(self, i):
        h = self.dt

        self.dx[:,i]
        self.v[:,i]

        k1 = self.acceleration(self.dx[:,i], self.v[:,i])

        self.v[:,i + 1] = self.v[:,i] + k1 * h / 2
        
        k2 = self.acceleration(self.dx[:,i], self.v[:,i + 1])
        
        self.v[:,i + 1] = self.v[:,i] + k2 * h / 2

        k3 = self.acceleration(self.dx[:,i], self.v[:,i + 1])
        
        self.v[:,i + 1] = self.v[:,i] + k3 * h

        k4 = self.acceleration(self.dx[:,i], self.v[:,i + 1])
        
        self.a[:,i + 1] = (k1 + 2*k2 + 2*k3 + k4) / 6.0
        self.v[:,i + 1] = self.v[:,i] + self.a[:,i + 1] * h
        self.x[:,i + 1] = self.x[:,i] + self.v[:, i + 1] * h
        
        # self.x[:,i + 1]      = self.x[:,i + 1] % self.L

        self.dx[:,i + 1]   = self.headway(self.x[:,i + 1])

    def acceleration(self, dx, v):
        return self.sensitivity * (self.V(dx) - v)