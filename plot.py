import numpy as np
from bodies import SpringDamper, Pendulum, CartPole
from mpc import MPC
import matplotlib.pyplot as plt
import sfml as sf

window = None

ti = 0.
tf = 5.
dt = 0.02

mpc_horizon = 5

time = np.arange(start=ti, stop=tf, step=dt, dtype=np.float64)


spring_damper = SpringDamper(window, pos_x=0.1, pos_y=0.5)
spring_damper.set_properties(k=10., m=1., c=0.1, x0=0.5)
spring_damper.integration_type = spring_damper.RUNGE_KUTTA

pendulum = Pendulum(window)
pendulum.set_properties(m=1.0, g=-9.8, L=0.25, k=0.1)
pendulum.set_position(0.5, 0.5)
pendulum.set_rotation(np.pi * 0.95)
pendulum.integration_type = pendulum.RUNGE_KUTTA
pendulum.nonlinear_mode = True

cart = CartPole(window)
cart.set_properties(m=0.2, M=0.5, I=0.006, g=-9.8, l=0.25, b=0.5)
cart.set_position(0.5, 0.5)
cart.set_rotation(np.pi * 0.99)
cart.nonlinear_mode = True
cart.integration_type = cart.RUNGE_KUTTA

mpc = MPC(dt, mpc_horizon)


# This works for springdamper and pendulum
R = np.matrix(np.eye(2) * 1e-5)
Q = np.matrix(np.matrix([[100, 0], [0.0, 1.0]]))
T = np.matrix(np.eye(2) * 1e-5)
mpc.set_cost_weights(Q, R, T)

mpc.Y_prime = np.matrix(np.zeros(shape=(2, mpc_horizon)))
mpc.Y_prime[0, :] = 0.5
mpc.C = np.matrix(np.eye(2))

mpc.set_body(spring_damper)

Xs = np.zeros(shape=(mpc.body.get_state().shape[0], time.shape[0]))
Us = np.zeros(shape=(mpc.body.u.shape[0],time.shape[0]))
Ys = np.zeros(shape=Xs.shape)
Y_desired = np.zeros(shape=Xs.shape)

for i, t in enumerate(time):
    if i % mpc_horizon == 0:
        mpc.calc_control()

    u = mpc.U[:, i % mpc_horizon]
    mpc.body.simulate(dt, np.matrix(u).transpose())

    Xs[:, i] = np.ravel(mpc.body.get_state())
    Us[:, i] = u
    Ys[:, i] = np.ravel(mpc.Y[:, i % mpc_horizon])

    Y_desired[:, i] = np.ravel(mpc.Y_prime[:, i % mpc_horizon])

fig, axarr = plt.subplots(Xs.shape[0] + 1, sharex=True)
for i in range(Xs.shape[0]):
    axarr[i].plot(time, Xs[i, :], label='Actual')
    axarr[i].plot(time, Ys[i, :], label='Predicted')
    axarr[i].plot(time, Y_desired[i, :], 'r--' , label='Desired')
    axarr[i].set_ylabel(mpc.body.state[i].name)
    axarr[i].legend()

axarr[1].set_xlabel('Time (s)')
axarr[0].set_title('State')

axarr[-1].plot(time, Us[1, :], label='Command')
axarr[-1].plot(time, np.zeros(shape=time.shape), 'r--' )
axarr[-1].set_ylabel('Command')

# fig = plt.figure()
# plt.plot(time, Us[1, :])
# plt.title('Command')
# plt.xlabel('Time (s)')
