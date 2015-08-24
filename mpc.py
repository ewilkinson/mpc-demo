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

        self.Xs = None  # state space for optimization

        self.C = None

        self.horizon = horizon
        self.dt = dt

    def cost_fn(self, u):
        cost_fn_u = np.matrix(u.reshape(self.U.shape))

        sum_cost = 0
        for i in range(self.horizon):
            tmp_u = cost_fn_u[:, i]

            X, X_prime = self.body.integrate(self.Xs[:, i], self.dt, tmp_u)
            self.Xs[:, i+1] = X

            # for debug so I can check each variable separately
            DY = self.Y_prime[:, i] - self.C * X

            a = DY.transpose() * self.Q * DY
            b = tmp_u.transpose() * self.R * tmp_u

            if i == 0:
                DU = tmp_u - self.last_u
            else:
                DU = tmp_u - cost_fn_u[:, i-1]
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

        self.last_u = np.matrix(np.zeros(shape=(body.B.shape[1], 1)))

        self.U = np.zeros(shape=(body.B.shape[1], self.horizon))
        self.Y = np.zeros(shape=(body.A.shape[1], self.horizon))

        self.Xs = np.matrix(np.zeros(shape=(body.A.shape[1], self.horizon + 1)))

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
        self.Xs[:, 0] = np.copy(self.body.get_state())

        solution = minimize(self.cost_fn,
                            x0=np.ravel(self.U),
                            constraints=self.body.cons,
                            method='SLSQP',
                            options={'eps': 1e-5})

        self.U = np.matrix(solution['x'].reshape(self.U.shape))

        self.last_u = self.U[:, -1]
