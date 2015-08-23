import numpy as np
from bodies import AbstractBody
from scipy.optimize import minimize


class MPC(object):
    def __init__(self, dt, horizon):
        self.body = None
        self.Q = None
        self.R = None

        self.U = None  # control commands
        self.Y = None  # predicted path
        self.Y_prime = None  # reference path

        self.horizon = horizon
        self.dt = dt

    def cost_fn(self, u):
        u = np.matrix(u).transpose()
        X = np.copy(self.X_iter)

        sum_cost = 0
        for i in range(2):
            temp_u = (np.matrix(np.zeros(u.shape)), u)[i == 0]  # if / else statement
            X, X_prime = self.body.integrate(X, self.dt, temp_u)

            # for debug so I can check each variable separately
            DY = self.y_prime - self.C * X
            DU = self.u - u
            a = DY.transpose() * self.Q * DY
            b = u.transpose() * self.R * u
            d = DU.transpose() * self.T * DU
            c = np.sum(a + b + d)
            sum_cost += c

        return sum_cost

    def set_body(self, body):
        """
        :type body: AbstractBody
        :param body: body which is to be controlled
        """

        self.body = body

        self.u = np.matrix(np.zeros(shape=(body.B.shape[1], 1)))

        self.U = np.zeros(shape=(self.u.shape[0], self.horizon))
        self.Y = np.zeros(shape=(body.A.shape[1], self.horizon))

    def set_cost_weights(self, Q, R, T=None):
        """
        :type Q: numpy.matrixlib.defmatrix.matrix
        :param Q: Cost matrix for observables

        :type R: numpy.matrixlib.defmatrix.matrix
        :param R: Cost matrix for control

        :type T: numpy.matrixlib.defmatrix.matrix
        :param T: Cost matrix for control
        """
        if type(Q) is not np.matrix or type(R) is not np.matrix:
            raise RuntimeError('Q and R must be numpy matrices')

        self.Q = Q
        self.R = R
        self.T = T

    def calc_control(self):
        self.X_iter = np.copy(self.body.get_state())
        self.U_iter = np.copy(self.U[:, -1])

        for i in range(self.horizon):
            self.y_prime = self.Y_prime[:, i]
            solution = minimize(self.cost_fn, self.u, constraints=self.body.cons, method='SLSQP', options={'eps': 1e-8})
            u = np.matrix(solution['x']).transpose()

            self.X_iter, X_prime = self.body.integrate(self.X_iter, self.dt, u)

            self.Y[:, i] = np.ravel(self.X_iter)
            self.U[:, i] = np.ravel(u)
            self.u = u
