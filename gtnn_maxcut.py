import numpy as np
import matplotlib.pylab as plt
import matplotlib
from gtnn_config import arg_list
from gtnn_object import GTNN
import torch
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

neuron = 800

arg_list['NUM_NEURON'] = neuron
arg_list['QFILE'] = './data/graphs/G15.txt'
arg_list['VMAX'] = 1
arg_list['DT'] = 0.001
arg_list['TMAX'] = 5
arg_list['TAU'] = 0.01
arg_list['ETA'] = 0.002
arg_list['LAMBDA'] = 10
arg_list['C'] = 1
arg_list['VTH'] = 0

myGTNN = GTNN()
myGTNN.init_v('None')
myGTNN.init_Q('user data')

norm = matplotlib.colors.Normalize()
fig = plt.figure()
fig.tight_layout(pad=3)

digit_ax = fig.add_subplot(111)
digit_ax.imshow(myGTNN.Q.numpy(), norm=norm, cmap=plt.cm.hot)
digit_ax.axis('off')
plt.show()

myGTNN.init_mask()
arg_list['TMAX'] = 5
myGTNN.learn = 0
b_temp = np.zeros((neuron, 1))
myGTNN.update(b = b_temp.reshape(neuron,1))

myGTNN.dimension_check()
myGTNN.run()
myGTNN.plot_general()