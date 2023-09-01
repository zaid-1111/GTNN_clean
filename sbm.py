import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from gtnn_config import arg_list
from gtnn_object import GTNN, get_num_neurons, max_cut
import math

# default_neuron = 800

# arg_list['NUM_NEURON'] = default_neuron
# arg_list['QFILE'] = './data/graphs/G14.txt'
# arg_list['VMAX'] = 1
# arg_list['DT'] = 0.001
# arg_list['TMAX'] = 50
# arg_list['TAU'] = 0.01
# arg_list['ETA'] = 0.002
# arg_list['LAMBDA'] = 10
# arg_list['C'] = 1
# arg_list['VTH'] = 0

# # Get the neurons number from file first
# neuron = get_num_neurons(arg_list['QFILE'])
# arg_list['NUM_NEURON'] = neuron
# # Run in max cut mode
# en_maxcut = 1
# # GTNN Initialization
# myGTNN = GTNN()
# myGTNN.init_v('None')
# myGTNN.init_Q('user data')

# Q_full = myGTNN.Q.numpy()
# # Q = np.triu(Q_full)
# Q = -Q_full.copy()
np.random.seed(42)

neuron = 2
Q = np.array([[0, 1],[1, 0]])


# x = 2 * np.random.rand(neuron, 1) - 1
# y = 2 * np.random.rand(neuron, 1) - 1
x = 2 * np.ones((neuron, 1)) - 1
y = 2 * np.ones((neuron, 1)) - 1

x *= 0.1
y *= 0.1

# x *= 0
# y *= 0

dt = 0.01
T = 10
duration = int(T/dt)

beta = 0.5
p = lambda x: 1
K = 1
# eta = 0.7*beta/(neuron**0.5)
eta = 0.1

x_record = np.zeros((neuron, duration))
Hsb_record = np.zeros(duration)

m = np.zeros_like(x)

for i in range(duration):
    # x = x + beta * y * dt
    # y = y - (K*x**3 + (beta - p(i*dt))*x - eta*Q@x) * dt
    m = -(beta*(K*x**3 + (beta - p(i*dt))*x + eta*Q@x) - m)
    x = x + m * dt

    x_record[:, i] = x.reshape(-1)
    Hsb = beta/2*np.sum(y**2) + K/4*np.sum(x**4) + (beta-p(i*dt))/2*np.sum(x**2) - eta/2* x.T @ Q @ x
    Hsb_record[i] = Hsb
    # if i % 100 == 0:
    #     num_maxcut, num_converged = max_cut(Q_full, x)
    #     print(f'iter: {i}; # cuts: {num_maxcut}')


norm = matplotlib.colors.Normalize()
fig = plt.figure()
fig.tight_layout(pad=5)
v_ax = fig.add_subplot(2, 1, 1)
vnorm_ax = fig.add_subplot(2, 1, 2)

vnorm_ax.plot(Hsb_record.reshape(-1))
# vnorm_ax.set_title('Hsb')
vnorm_ax.set_ylabel('Hsb')
vnorm_ax.set_xlabel('iter(n)')

# v_temp = self.vp_ev.T - self.vn_ev.T
v_ax.plot(x_record.T)
# v_ax.set_title('Evolution of Membrane Potential')
v_ax.set_ylabel('x')
plt.show()
