import numpy as np
import matplotlib.pylab as plt
import matplotlib
import torch
from gtnn_config import arg_list
from gtnn_object import GTNN, get_num_neurons

neuron = 2

arg_list['NUM_NEURON'] = neuron
arg_list['QFILE'] = './data/graphs/G14.txt'
arg_list['VMAX'] = 1
arg_list['DT'] = 0.001
arg_list['TMAX'] = 10
arg_list['TAU'] = 0.01
arg_list['ETA'] = 0.002
arg_list['LAMBDA'] = 20
arg_list['C'] = 1
arg_list['VTH'] = 0

Q = np.array([[0.0, 1.0], [1.0, 0.0]])
# Get the neurons number from file first
# neuron = get_num_neurons(arg_list['QFILE'])
# arg_list['NUM_NEURON'] = neuron
# Run in max cut mode
en_maxcut = 1
# GTNN Initialization
myGTNN = GTNN()
myGTNN.init_v('random')
np.save('vp.npy', myGTNN.vp.numpy())
np.save('vn.npy', myGTNN.vn.numpy())
# myGTNN.init_Q('user data')
myGTNN.Q = torch.tensor(Q, dtype=torch.float32)
myGTNN.init_mask()
# Plot the Q/adjacency matrix
# myGTNN.plot_adjacency()
# No Q adaptation
myGTNN.learn = 0
# Zero inputs
b_temp = np.zeros((neuron, 1), dtype=np.float32)
myGTNN.update(b = b_temp.reshape(neuron,1))
# Sanity check before run
myGTNN.dimension_check()
myGTNN.run(0)
myGTNN.report_maxcut()
myGTNN.report_time()
myGTNN.plot_general()