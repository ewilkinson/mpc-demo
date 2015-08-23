import sfml as sf
import numpy as np

class Param(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __str__(self):
        return self.name + ": %.3f" % self.value

    __repr__ = __str__

    def __mul__(self, other):
        if type(other) is not Param:
            return self.value * other
        else:
            return self.value * other.value

    __rmul__ = __mul__

    def __abs__(self):
        self.value = abs(self.value)

    def __add__(self, other):
        if type(other) is not Param:
            return self.value + other
        else:
            return self.value + other.value

    def __sub__(self, other):
        if type(other) is not Param:
            return self.value - other
        else:
            return self.value - other.value

    def __div__(self, other):
        if type(other) is not Param:
            return self.value / other
        else:
            return self.value / other.value

    def __rdiv__(self, other):
        if type(other) is not Param:
            return other / self.value
        else:
            return other.value / self.value

    def __eq__(self, other):
        if type(other) is not Param:
            return self.value == other
        else:
            return self.value == other.value

    @staticmethod
    def list_to_matrix(state):
        matrix = np.matrix(np.zeros(shape=(len(state), 1)))
        for i, var in enumerate(state):
            matrix[i,0] = var.value
        return matrix


class AbstractBody(object):
    """
    A class with common attributes for simulation bodies.
    Contains useful features for dealing with SFML display
    """

    EULER = 'EULER'
    RUNGE_KUTTA = 'RK'

    def __init__(self, window):
        self.window = window

        if window is not None:
            self.window_x = window.size[0]
            self.window_y = window.size[1]

        self.EULER = AbstractBody.EULER
        self.RUNGE_KUTTA = AbstractBody.RUNGE_KUTTA

        self.renderables = []
        self.state = []

        self.nonlinear_mode = False
        self.integration_type = self.EULER

        self.cons = ()

    def _degree_to_rad(self, degree):
        return degree * np.pi / 180.0

    def _rad_to_degree(self, rad):
        return rad / np.pi * 180

    def render(self):
        for polygon in self.renderables:
            self.window.draw(polygon)

    def simulate(self):
        raise NotImplementedError("This method needs to be implemented")

    def set_rotation(self, theta):
        for body in self.renderables:
            body.rotation = -self._rad_to_degree(theta) + 90

    def set_position(self, x, y):
        for body in self.renderables:
            body.position = sf.Vector2(int(x * self.window_x), int(y * self.window_y))

    def get_state(self):
        """
        :rtype: np.matrix
        :return:
        """
        return Param.list_to_matrix(self.state)

    def set_state(self, X):
        for i in range(X.shape[0]):
            self.state[i].value = X[i, 0]

    def integrate(self, X, dt, u=0):
        # Euler
        if self.integration_type == self.EULER:
            X_prime = self.A * X + self.B * u

        # Runge-Kutta
        elif self.integration_type == self.RUNGE_KUTTA:
            X_k1 = self.A * X + self.B * u
            X_k2 = self.A * (X + dt / 2 * X_k1) + self.B * u
            X_k3 = self.A * (X + dt / 2 * X_k2) + self.B * u
            X_k4 = self.A * (X + dt * X_k3) + self.B * u

            X_prime = 1 / 6.0 * (X_k1 + 2 * X_k2 + 2 * X_k3 + X_k4)
        else:
            raise ValueError('Integration Type Not Recognized : %s' % self.integration_type)

        return X + X_prime * dt, X_prime

    def add_max_value_constraint(self, indx, value):
        self.cons += ({'type': 'ineq',
                      'fun': lambda x: np.array([-abs(x[indx]) + value]),  # max pendulum torque
                      'jac': lambda x: np.array([0.0, -1.0]) if x[1] > 0 else np.array([0.0, 1.0])},)


class SpringDamper(AbstractBody):
    def __init__(self, window, pos_x=0.0, pos_y=0.0):
        super(SpringDamper, self).__init__(window)

        self.pos_x = Param("x", pos_x * 1.)
        self.pos_y = Param("y", pos_y * 1.)
        self.pos_x_d = Param("x_dot", 0.0)
        self.pos_y_d = Param("y_dot", 0.0)

        self.state = [self.pos_x, self.pos_x_d]

        self.poly_radius = 10

        if window is not None:
            self.renderables = self._create_renderables()

        self.add_max_value_constraint(1, 20)

    def set_properties(self, k, m, c, x0):
        self.k = k
        self.m = m
        self.c = c
        self.x0 = x0

        self.A = np.matrix([[0, 1], [-self.k / self.m * (1 - self.x0 / (self.pos_x + 1e-5)), -self.c / self.m]])

        self.B = np.matrix(np.zeros(shape=(2, 2)))
        self.B[1, 1] =1 / self.m

        self.u = np.matrix(np.zeros(shape=(self.B.shape[1], 1)))

    def _create_renderables(self):
        self.polygon = sf.CircleShape(self.poly_radius)
        self.polygon.fill_color = sf.Color.BLACK
        self.polygon.position = sf.Vector2(int(self.pos_x * self.window_x), int(self.pos_y * self.window_y))
        self.polygon.origin = sf.Vector2(self.poly_radius, self.poly_radius)
        return [self.polygon]

    def set_position(self, pos_x, pos_y):
        super(SpringDamper, self).set_position(pos_x, pos_y)

    def simulate(self, dt, u=0):
        x1 = self.state[0].value
        self.A[1, 0] = -self.k / self.m * (1 - self.x0 / (x1 + 1e-7))

        X, X_prime = self.integrate(self.get_state(), dt, u)

        self.set_position(X[0,0], self.pos_y.value)
        self.set_state(X)

    def set_state(self, X):
        super(SpringDamper, self).set_state(X)

        x1 = self.state[0].value
        self.A[1, 0] = -self.k / self.m * (1 - self.x0 / (x1 + 1e-7))


class Pendulum(AbstractBody):
    def __init__(self, window, theta=0.0):
        super(Pendulum, self).__init__(window)

        self.theta = Param('theta', theta)
        self.theta_d = Param('theta_d', 0.)
        self.state = [self.theta, self.theta_d]

        self.circle_radius = 20

        self.cons = ({'type': 'ineq',
                      'fun': lambda x: np.array([-abs(x[1]) + 10]),  # max pendulum torque
                      'jac': lambda x: np.array([0.0, -1.0]) if x[1] > 0 else np.array([0.0, 1.0])})

    def set_properties(self, m, g, L, k):
        """
        :param m: mass
        :param g: gravity
        :param L: rod length
        :param k: damping
        """
        self.m = m
        self.g = g
        self.L = L
        self.k = k

        self.A = np.matrix([[0,                            1],
                            [0, - self.k / (self.m * self.L)]])

        self.B = np.matrix(np.zeros(shape=(2, 2)))
        self.B[1, 1] = 1.0 / (self.m * self.L)

        self.u = np.matrix(np.zeros(shape=(self.B.shape[1], 1)))

        if self.window is not None:
            self.renderables = self._create_renderables()

    def _create_renderables(self):
        size_x = int(self.L * self.window_x)
        size_y = 25
        rod = sf.RectangleShape(size=sf.Vector2(size_x, size_y))
        rod.fill_color = sf.Color.BLACK
        rod.origin = sf.Vector2(0, int(size_y / 2))

        weight = sf.CircleShape(self.circle_radius)
        weight.fill_color = sf.Color.BLUE
        weight.origin = sf.Vector2(self.circle_radius - size_x, self.circle_radius)

        pivot = sf.CircleShape(int(size_y / 2))
        pivot.fill_color = rod.fill_color
        pivot.origin = sf.Vector2(pivot.radius, pivot.radius)
        return [rod, weight, pivot]

    def set_rotation(self, theta):
        """
        Rotation must be in radians!
        :param theta: radians
        """
        super(Pendulum, self).set_rotation(theta)

        self.theta.value = theta

    def set_velocity(self, theta_d):
        self.theta_d.value = theta_d

    def simulate(self, dt, u=0):

        if self.nonlinear_mode:
            x1 = self.state[0].value
            self.A[1, 0] = (self.g / self.L) * np.sin(x1) / (x1 + 1e-7)
        else:
            self.A[1,0] = self.g/self.L

        X, X_prime = self.integrate(self.get_state(), dt, u)

        self.set_rotation(theta=X[0,0])
        self.set_velocity(theta_d=X[1,0])


class CartPole(AbstractBody):
    def __init__(self, window):
        super(CartPole, self).__init__(window)

        self.pos_x = Param("x", 0.0)
        self.pos_x_d = Param("x_dot", 0.0)
        self.theta = Param('theta', 0.0)
        self.theta_d = Param('theta_d', 0.)

        self.render_y = 0.0

        self.state = [self.pos_x, self.pos_x_d, self.theta, self.theta_d]

        self.circle_radius = 20

        self.cons = ()

    def set_properties(self, m, M, I, g, l, b):
        """
        :param m: mass of pendulum
        :param M: mass of cart
        :param I: moment of inertia
        :param g: gravity
        :param L: rod length
        :param k: damping
        """
        self.m = m
        self.M = M
        self.I = I
        self.g = g
        self.l = l
        self.b = b

        # from http://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=ControlStateSpace
        # and http://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=SystemModeling
        p = I*(M+m)+M*m*(l**2)
        self.p = p

        self.A = np.matrix([[0,                 1,                   0,      0],
                            [0,   -(I+m*l**2)*b/p,     (m**2*g*l**2)/p,      0],
                            [0,                 0,                   0,      1],
                            [0,        -(m*l*b)/p,       m*g*l*(M+m)/p,      0]])

        self.B = np.matrix(np.zeros(shape=(4, 4)))
        self.B[1, 1] = (I + m * l ** 2) / p
        self.B[3, 3] = m * l / p

        self.u = np.matrix(np.zeros(shape=(self.B.shape[1], 1)))

        if self.window is not None:
            self.renderables = self._create_renderables()

    def _create_renderables(self):
        size_x = int(self.l * self.window_x)
        size_y = 25
        rod = sf.RectangleShape(size=sf.Vector2(size_x, size_y))
        rod.fill_color = sf.Color.BLACK
        rod.origin = sf.Vector2(0, int(size_y / 2))

        weight = sf.CircleShape(self.circle_radius)
        weight.fill_color = sf.Color.BLUE
        weight.origin = sf.Vector2(self.circle_radius - size_x, self.circle_radius)

        pivot = sf.CircleShape(int(size_y / 2))
        pivot.fill_color = rod.fill_color
        pivot.origin = sf.Vector2(pivot.radius, pivot.radius)

        cart = sf.RectangleShape(size=sf.Vector2(size_x, 75))
        cart.fill_color = sf.Color.GREEN
        cart.origin = sf.Vector2(int(size_x/2), 0)
        return [cart, rod, weight, pivot]

    def set_position(self, pos_x=0.5, pos_y=0.5):
        self.pos_x.value = pos_x * 1.
        self.render_y = pos_y
        super(CartPole, self).set_position(pos_x, pos_y)

    def set_velocity(self, pos_x_d, pos_y_d):
        self.pos_x_d.value = pos_x_d

    def simulate(self, dt, u=0):
        if self.nonlinear_mode:
            theta = self.state[2].value
            self.A[3, 2] = self.m*self.g*self.l*(self.M+self.m)/self.p *  np.sin(theta) / (theta + 1e-7)
            self.A[1, 2] = (self.m**2* self.g*self.l**2)/ self.p * np.sin(theta) / (theta + 1e-7)

        X, X_prime = self.integrate(self.get_state(), dt, u)

        for i, s in enumerate(self.state):
            s.value = X[i, 0]

        self.set_position(X[0,0], self.render_y)
        self.set_velocity(X[1,0], 0.0)
        self.set_rotation(X[2,0])

    def set_rotation(self, theta):
        self.theta.value = theta

        for body in self.renderables[1:]:
            body.rotation = -self._rad_to_degree(theta) + 90


if __name__ == '__main__':
    window_x = 800
    window_y = 800
    window = sf.RenderWindow(sf.VideoMode(window_x, window_y), "PySFML test")
    window.position = sf.Vector2(10, 50)
    window.title = 'MPC window'
    window.framerate_limit = 60

    g = SpringDamper(window)
    print g.state

    a = Param('a', 2.0)
    b = Param('b', 1.0)
    print a, b
    print 'A - B : ', a - b
    print 'A * B : ', a * b
    print 'B * A : ', b * a
    print 'A / B : ', a / b
    print 'B / A : ', b / a

    # import time
    #
    # for i in range(360):
    #     window.clear(sf.Color.WHITE)
    #     p = Pendulum(window)
    #     p.set_properties(1.0, 1.0, 0.25, 1.0)
    #     p.set_position(0.5, 0.5)
    #     p.set_rotation(p._degree_to_rad(i))
    #     p.render()
    #     window.display()
    #     time.sleep(0.05)

    cart = CartPole(window)
    import time

    for i in range(360):
        window.clear(sf.Color.WHITE)
        cart.set_properties(m=0.2, M=0.5, I=0.006, g= 9.8, l=0.25, b=0.1)
        cart.set_position(0.5, 0.5)
        cart.set_rotation(cart._degree_to_rad(i))
        cart.render()
        window.display()
        time.sleep(0.05)

