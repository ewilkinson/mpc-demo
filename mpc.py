import numpy as np
from bodies import AbstractBody
from scipy.optimize import minimize

class MPC(object):
    def __init__(self, dt, horizon):
        self.A = None
        self.B = None
        self.Q = None
        self.R = None

        self.X = None  # current state

        self.U = None  # control commands
        self.Y = None  # predicted path
        self.Y_prime = None  # reference path

        self.horizon = horizon
        self.dt = dt

        self.integration_type = AbstractBody.EULER

        self.cons = ()

    def set_constraints(self, cons):
        """
        Set the constraints for the scipy minimize function
        http://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html

        :type cons: tuple
        :param cons: constraints are defined as a sequence of dictionaries, with keys type, fun and jac.
        """
        self.cons = cons

    def current_state(self, X):
        self.X = X

    def cost_fn(self, u):
        u = np.matrix(u).transpose()
        X = np.copy(self.X_iter)
        dt = self.dt

        if self.integration_type == AbstractBody.EULER:
            X_prime = self.A * X + self.B * u
        elif self.integration_type == AbstractBody.RUNGE_KUTTA:
            X_k1 = self.A * X + self.B * u
            X_k2 = self.A * (X + dt/2 * X_k1) + self.B * u
            X_k3 = self.A * (X + dt/2 * X_k2) + self.B * u
            X_k4 = self.A * (X + dt * X_k1) + self.B * u

            X_prime = 1 / 6.0 * (X_k1 + 2* X_k2 + 2* X_k3 + X_k4)

        X = X + X_prime*dt

        DY = self.y_prime - self.C * X

        # for debug so I can check each variable separately
        a = DY.transpose() * self.Q  * DY
        b = u.transpose() * self.R * u
        c = np.sum(a + b)
        return c

    def set_model(self, A, B, C):
        if type(A) is not np.matrix or type(B) is not np.matrix or type(C) is not np.matrix:
            raise RuntimeError('A, B. and C must be numpy matrices')

        self.A = A
        self.B = B
        self.C = C

        self.u = np.matrix(np.zeros(shape=(B.shape[1],1)))
        self.U = np.zeros(shape=(B.shape[1], self.horizon))
        self.Y = np.zeros(shape=(A.shape[1], self.horizon))

    def set_cost_weights(self, Q, R):
        if type(Q) is not np.matrix or type(R) is not np.matrix:
            raise RuntimeError('Q and R must be numpy matrices')

        self.Q = Q
        self.R = R

    def _iterate(self):
        dt = self.dt
        self.X_iter = np.copy(self.X)

        for i in range(self.horizon):
            self.y_prime = self.Y_prime[:,i]
            solution = minimize(self.cost_fn, self.u, constraints=self.cons)

            u = np.matrix(solution['x']).transpose()
            X = self.X_iter

            if self.integration_type == AbstractBody.EULER:
                X_prime = self.A * X + self.B * u
            elif self.integration_type == AbstractBody.RUNGE_KUTTA:
                X_k1 = self.A * X + self.B * u
                X_k2 = self.A * (X + dt/2 * X_k1) + self.B * u
                X_k3 = self.A * (X + dt/2 * X_k2) + self.B * u
                X_k4 = self.A * (X + dt * X_k1) + self.B * u

                X_prime = 1 / 6.0 * (X_k1 + 2* X_k2 + 2* X_k3 + X_k4)


            self.X_iter = X + X_prime*dt

            self.Y[:,i] = np.ravel(self.X_iter)
            self.U[:,i] = np.ravel(u)
            self.u = u

