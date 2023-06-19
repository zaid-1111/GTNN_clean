from itertools import count
import gtnn_functions as gtnn
import numpy as np
import matplotlib.pylab as plt
import matplotlib
from scipy.sparse import csr_matrix, vstack, block_diag, coo_matrix

num_neuron = 2000

# layer_sizes = np.empty((0))
# # Q, _ = gtnn.genQ_random(0.2, 0, 1, 0.5, layer_sizes)
# Q = np.random.rand(num_neuron, num_neuron)
# M = np.random.rand(num_neuron, num_neuron)
# Q[M>0.1] = 0
# nnz = np.count_nonzero(Q, axis=1)
# Q *= 1/nnz.reshape(num_neuron, 1)
# Q *= np.logical_not(np.eye(num_neuron))
# print(nnz[0])

# norm = matplotlib.colors.Normalize()
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.imshow(Q, norm=norm, cmap=plt.cm.gray)
# # ax.set_title(titles[i], fontsize=10)
# ax.set_xticks([])
# ax.set_yticks([])
# plt.show()

# cooQ = coo_matrix(Q)
# print(cooQ.data.size)
# size = cooQ.data.size
# Qdata = np.concatenate((cooQ.data.reshape(size, 1), cooQ.row.reshape(size, 1), cooQ.col.reshape(size, 1)), axis=1)

# np.savetxt('sparse_random_Q.txt', Qdata)

num_layers = 16
layer_sizes = np.random.dirichlet(np.ones((num_layers), \
    dtype=np.single)*100, size=1).reshape(num_layers)
layer_sizes *= num_neuron
layer_sizes = np.rint(layer_sizes).astype(np.uint)
if np.sum(layer_sizes) != num_neuron:
    layer_sizes[-1] += num_neuron - np.sum(layer_sizes)


M = np.random.rand(num_layers, num_layers)
# Radom block
# M[M > 0.2] = 0
for i in range(num_layers):
    for j in range(num_layers):
        if i+j == 8 or i + j == 24 or num_layers-j+i == 8 or num_layers+j-i == 8:
            M[i, j] = 0
M = np.logical_not(M)
for i in range(num_layers):
    for j in range(num_layers):
        if (i+j == 12 or i + j == 20 or num_layers-j+i == 12 or num_layers+j-i == 12)\
            and i >= 4 and i <= 12 and j >= 4 and j <= 12:
            M[i, j] = 1

norm = matplotlib.colors.Normalize()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.imshow(M, norm=norm, cmap=plt.cm.gray)
# ax.set_title(titles[i], fontsize=10)
ax.set_xticks([])
ax.set_yticks([])
plt.show()

offset = np.cumsum(layer_sizes)
offset = np.insert(offset, 0, 0)
Mask = np.zeros((num_neuron, num_neuron))
for i in range(num_layers):
    for j in range(num_layers):
        if M[i, j]:
            Mask[offset[i]:offset[i+1], offset[j]:offset[j+1]] = 1

Q = np.random.rand(num_neuron, num_neuron)
Q *= Mask
Q[Q > 0.6] = 0
norm = matplotlib.colors.Normalize()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.imshow(Q, norm=norm, cmap=plt.cm.gray)
# ax.set_title(titles[i], fontsize=10)
ax.set_xticks([])
ax.set_yticks([])
plt.show()

cooQ = coo_matrix(Q)
print(cooQ.data.size)
size = cooQ.data.size
Qdata = np.concatenate((cooQ.data.reshape(size, 1), cooQ.row.reshape(size, 1), cooQ.col.reshape(size, 1)), axis=1)

np.savetxt('sparse_pattern1_Q.txt', Qdata)

