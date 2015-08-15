import sfml as sf
import numpy as np
from bodies import SpringDamper, Pendulum, CartPole
from mpc import MPC

# Create the main window
settings = sf.ContextSettings()
settings.antialiasing_level = 8  # doesn't seem to do anything

window_x = 800
window_y = 800
window = sf.RenderWindow(sf.VideoMode(window_x, window_y), "PySFML test", sf.Style.DEFAULT, settings)
window.position = sf.Vector2(10, 50)
window.title = 'MPC window'
window.framerate_limit = 60


def add_time(window, time):
    text = "Time : %.2f" % time
    sf_text = sf.Text(text)
    sf_text.color = sf.Color.BLACK
    window.draw(sf_text)


class System(object):
    def __init__(self, window, dt=0.01):
        self.window = window
        self.dt = dt
        self.time = 0
        self.mpc_horizon = 10

        spring_damper = SpringDamper(window, pos_x=0.2, pos_y=0.5)
        spring_damper.set_properties(k=10., m=1., c=0.1, x0=0.5)
        spring_damper.integration_type = spring_damper.RUNGE_KUTTA

        pendulum = Pendulum(window)
        pendulum.set_properties(m=1.0, g=-9.8, L=0.25, k=0.1)
        pendulum.set_position(0.5, 0.5)
        pendulum.set_rotation(np.pi * 0.5)
        pendulum.integration_type = pendulum.RUNGE_KUTTA
        pendulum.nonlinear_mode = True

        cart = CartPole(window)
        cart.set_properties(m=0.2, M=0.5, I=0.006, g=-9.8, l=0.25, b=0.5)
        cart.set_position(0.5, 0.5)
        cart.set_rotation(np.pi * 0.99)
        cart.nonlinear_mode = True
        cart.integration_type = cart.RUNGE_KUTTA

        self.mpc = MPC(dt, self.mpc_horizon)


        R = np.matrix(np.eye(2)*0.001)
        Q = np.matrix(np.matrix([[100,0],[0,0.1]]))
        self.mpc.set_cost_weights(Q, R)
        self.mpc.set_constraints(pendulum.cons)

        self.mpc.Y_prime = np.matrix(np.zeros(shape=(2, self.mpc_horizon)))
        self.mpc.Y_prime[0,:] = np.pi

        # self.bodies = [spring_damper, pendulum]
        self.bodies = [pendulum]

    def simulate(self):
        for body in self.bodies:
            self.mpc.set_model(body.A, body.B, np.matrix([[1,0]]))
            self.mpc.current_state(body.get_state())

            self.mpc._iterate()
            u = self.mpc.U[:,0]

            body.u = np.matrix(u).transpose()
            body.simulate(self.dt)

    def render(self):
        for body in self.bodies:
            body.render()

    def step(self):
        self.time += self.dt
        self.simulate()
        self.render()

# Start the game loop
paused = False
system = System(window=window, dt=0.01)

# center_radius = 12.5
# center = sf.CircleShape(center_radius)
# center.position = sf.Vector2(int(window_x/2), int(window_y/2))
# center.fill_color = sf.Color.BLACK
# center.origin = sf.Vector2(center_radius,center_radius)

while (window.is_open):
    window.clear(sf.Color.WHITE)

    for event in window.events:
        if type(event) == sf.CloseEvent:
            window.close()
        elif type(event) is sf.MouseButtonEvent:
            paused = not paused
        elif type(event) is sf.MouseMoveEvent:
            pos_x = event.position[0]
            pos_y = event.position[1]

    add_time(window, system.time)

    system.step()

    window.display()

print 'Finished!'