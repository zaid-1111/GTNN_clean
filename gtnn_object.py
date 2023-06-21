#  Washington University hereby grants to you a non-transferable,
#  non-exclusive, royalty-free, non-commercial, research license to use and
#  copy the computer code provided here (the �Software�).  You agree to
#  include this license and the above copyright notice in all copies of the
#  Software.  The Software may not be distributed, shared, or transferred to
#  any third party.  This license does not grant any rights or licenses to
#  any other patents, copyrights, or other forms of intellectual property
#  owned or controlled by Washington University.  If interested in obtaining
#  a commercial license, please contact Washington University's Office of
#  Technology Management (otm@dom.wustl.edu).
 
#  YOU AGREE THAT THE SOFTWARE PROVIDED HEREUNDER IS EXPERIMENTAL AND IS
#  PROVIDED �AS IS�, WITHOUT ANY WARRANTY OF ANY KIND, EXPRESSED OR IMPLIED,
#  INCLUDING WITHOUT LIMITATION WARRANTIES OF MERCHANTABILITY OR FITNESS FOR
#  ANY PARTICULAR PURPOSE, OR NON-INFRINGEMENT OF ANY THIRD-PARTY PATENT,
#  COPYRIGHT, OR ANY OTHER THIRD-PARTY RIGHT.  IN NO EVENT SHALL THE
#  CREATORS OF THE SOFTWARE OR WASHINGTON UNIVERSITY BE LIABLE FOR ANY
#  DIRECT, INDIRECT, SPECIAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF OR IN
#  ANY WAY CONNECTED WITH THE SOFTWARE, THE USE OF THE SOFTWARE, OR THIS
#  AGREEMENT, WHETHER IN BREACH OF CONTRACT, TORT OR OTHERWISE, EVEN IF SUCH
#  PARTY IS ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. YOU ALSO AGREE THAT
#  THIS SOFTWARE WILL NOT BE USED FOR CLINICAL PURPOSES.
from gtnn_config import arg_list
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import numpy.matlib as matlab
global neuron
global duration
from scipy.sparse import coo_matrix
import torch
import time
plt.rcParams.update({'font.size': 20})

def max_cut(adj, v):
    v = v.reshape(arg_list['NUM_NEURON'],)
    e = 1e-3
    v[v > arg_list['VMAX'] - e] = 1
    v[v < -arg_list['VMAX'] + e] = 0
    num_converged = np.sum(v == 1) + np.sum(v == 0)
    # 0 activation
    v = v > 0
    out = 2*np.outer(v, v)
    i = matlab.repmat(v, arg_list['NUM_NEURON'], 1).T
    j = matlab.repmat(v, arg_list['NUM_NEURON'], 1)
    # print(j.shape)
    # print(out.shape)
    # print(out)
    # print(j)
    # print(i)
    num_maxcut = np.sum(np.triu(adj * -(out-i-j)))

    return num_maxcut, num_converged

def get_num_neurons(user_data_file=None):
    user_data = np.loadtxt(user_data_file, max_rows=1, dtype=int)
    return user_data[0]


def genQ_txt(num_neuron=None, user_data_file=None):
    global arg_list

    user_data = np.loadtxt(user_data_file, skiprows=1, dtype=int)
    col = user_data[:,1].T - 1
    row = user_data[:,0].T - 1
    data = user_data[:,2].T
    coo_Q = coo_matrix((data,(row,col)), shape=(num_neuron, num_neuron), dtype=int)
    array_Q = coo_Q.toarray()
    
    full_Q = (np.triu(array_Q, k=1))
    full_Q = full_Q + full_Q.T
    

    return torch.tensor(full_Q, dtype=torch.double)

class GTNN:
    # Vanilla init for GTNN object
    def __init__(self):
        global neuron, duration
        neuron = arg_list['NUM_NEURON']
        duration = int(arg_list['TMAX']/arg_list['DT'])
        self.vp = torch.empty((neuron, 1))
        self.vn = torch.empty((neuron, 1))
        self.Psip = torch.zeros((neuron, 1))
        self.Psin = torch.zeros((neuron, 1))
        self.b = torch.empty(0)
        self.mask = torch.empty((neuron, neuron))
        self.Q = torch.empty((neuron, neuron))
        self.vp_ev = np.empty((neuron, duration))
        self.vn_ev = np.empty((neuron, duration))
        self.true_spikes = np.empty((neuron, duration))
        self.spike_rate = np.empty((neuron, duration))
        self.learn = 0
        
        self.synapse_obj = None
        self.pulse_m = 1
        self.pulse_w = arg_list['DT']
    
    def dimension_check(self):
        assert self.b.shape == (neuron, 1) or self.b.shape == (neuron, duration)
        assert self.mask.shape == (neuron, neuron)
        assert self.Q.shape == (neuron, neuron)

    # Public: Refresh the record on membrane potential and spiking activity
    # Call after reset the TMAX parameter for the *Next* run
    def refresh_record(self):
        global neuron, duration
        self.vp_ev = np.empty((neuron, duration))
        self.vn_ev = np.empty((neuron, duration))
        self.true_spikes = np.empty((neuron, duration))
        self.spike_rate = np.empty((neuron, duration))
    
    # 'uniform' -0.2
    # 'random' uniformly random [0, -0.2]
    def init_v(self, mode):
        global neuron, duration
        self.vp = torch.rand(neuron, 1, dtype=torch.double) * -0.5 
        self.vn = torch.rand(neuron, 1, dtype=torch.double) * -0.5 

        pass

    def init_b(self, mode):
        global neuron, duration
        if np.char.equal(mode, 'DC'):
            self.b = 0.6*np.random.rand(neuron, 1)-0.3
        pass

    def init_Q(self, mode):
        global neuron, duration

        if np.char.equal(mode, "user data"):
            num_neuron = arg_list['NUM_NEURON']
            Qfile = arg_list['QFILE']
            self.Q = genQ_txt(num_neuron=num_neuron, user_data_file=Qfile)
        else:
            self.Q = 1/neuron * (np.random.rand(neuron, neuron)-0.5)*np.logical_not(np.eye(neuron))
            self.Q = torch.tensor(self.Q)
    
    def init_mask(self):
        global neuron, duration
        self.mask = np.logical_not(np.eye(neuron))
        pass

    # Private: update Q
    def updateQ(self):
        threshold = 1e-4
        Qgrad = np.outer((self.Psip - self.Psin), (self.vp - self.vn))
        deltaQ=0.5 * arg_list['ETA'] * self.mask * Qgrad
        deltaQ[np.abs(deltaQ)<threshold] = 0
        if self.synapse_obj == None:
            self.Q += deltaQ
        else:
            deltaQ = torch.tensor(deltaQ)
            self.synapse_obj.synapse_evolve(-self.pulse_m*deltaQ, self.pulse_w)
            self.Q = self.synapse_obj.read()
        pass
    
    # def updateQ_FN(self):
    #     threshold = 1e-4
    #     Qgrad = np.outer((self.Psip - self.Psin), (self.vp - self.vn))
    #     # S=self.pulse_m
    #     # pw=self.pulse_w
    #     deltaQ=0.5 * arg_list['ETA'] * self.mask * Qgrad
    #     deltaQ[np.abs(deltaQ)<threshold] = 0
    #     deltaQ = torch.tensor(deltaQ)
    #     self.synapse_obj.cont_update(-self.pulse_m*deltaQ ,self.pulse_w)
    #     self.Q=self.synapse_obj.y.numpy()
    #     pass
    
    # Public: Update inputs or mask before next run
    def update(self, b = np.empty((0)), M = np.empty((0))):
        global neuron, duration
        duration = int(arg_list['TMAX']/arg_list['DT'])
        if len(b) != 0:
            if not isinstance(b, torch.Tensor):
                self.b = torch.tensor(b)
        if len(M) != 0:
            if not isinstance(M, torch.Tensor):
                self.M = torch.tensor(M)
        assert self.b.shape == (neuron, 1) or self.b.shape == (neuron, duration)
        assert self.mask.shape == (neuron, neuron)
    
    def refresh_vpvn(self):
        global neuron, duration
        self.vp *= 0
        self.vn *= 0
        self.Psin *= 0
        self.Psip *= 0
            
    # Public: run with current vp, vn, b, and Q, for Tmax time
    def run(self, flag):
        global neuron, duration
        self.start_time = time.time()
        window_size = 100
        spike_rate_window = np.zeros((neuron, window_size))
        #
        continuous_b = self.b.shape[1] == duration
        # DC input
        if not continuous_b:
            b_iter = self.b.reshape(neuron, 1)
        for iter in range(duration):
            # Record membrane potential
            self.vp_ev[:, iter] = self.vp.numpy().reshape(neuron,)
            self.vn_ev[:, iter] = self.vn.numpy().reshape(neuron,)
            # Calculate gradient
            Qv = torch.matmul(self.Q, (self.vp-self.vn))
            # AC input
            if continuous_b:
                b_iter = self.b[:, iter].reshape(neuron, 1)

            Gp = self.vp - b_iter + Qv + self.Psip
            Gn = self.vn + b_iter - Qv + self.Psin
            # Reset spikes
            self.Psip *= 0
            self.Psin *= 0
            # GT dynamics
            self.vp = self.vp + (arg_list['DT']/arg_list['TAU']) * ((self.vp*self.vp - arg_list['VMAX']**2) * Gp)\
                    / (-self.vp * Gp + arg_list['LAMBDA'] * arg_list['VMAX'])
            self.vn = self.vn + (arg_list['DT']/arg_list['TAU']) * ((self.vn*self.vn - arg_list['VMAX']**2) * Gn)\
                    / (-self.vn * Gn + arg_list['LAMBDA'] * arg_list['VMAX'])
            
            # vp = vp - arg_list['TAU'] * (vp + np.sign(Gp) * arg_list['VMAX'])
            # vn = vn - arg_list['TAU'] * (vn + np.sign(Gn) * arg_list['VMAX'])


            # Sanity check
            if not flag:
                vp_flag = torch.logical_or(torch.abs(self.vp) > arg_list['VMAX'], torch.abs(Gp) > arg_list['LAMBDA'])
                vn_flag = torch.logical_or(torch.abs(self.vn) > arg_list['VMAX'], torch.abs(Gn) > arg_list['LAMBDA'])
                if torch.any(vp_flag) or torch.any(vn_flag):
                    print("GT Sanity Check %d" %(iter))
                    print("Gp: %d", torch.sum(torch.abs(Gp) > arg_list['LAMBDA']))
                    print("Gn: %d", torch.sum(torch.abs(Gn) > arg_list['LAMBDA']))
                    break
            
            # Generate spikes
            self.Psip[self.vp>arg_list['VTH']] = arg_list['C']
            self.Psin[self.vn>arg_list['VTH']] = arg_list['C']
            self.vp = self.vp*(self.vp<=arg_list['VTH']) + arg_list['VTH']*(self.vp>arg_list['VTH'])
            self.vn = self.vn*(self.vn<=arg_list['VTH']) + arg_list['VTH']*(self.vn>arg_list['VTH'])

            self.true_spikes[:, iter] = (self.Psip + self.Psin).reshape(neuron,).numpy() * iter
            spike_rate_window[:, int(iter%window_size)] = (self.Psip + self.Psin).reshape(neuron,).numpy()
            if iter < window_size:
                self.spike_rate[:, iter] = np.sum(spike_rate_window, axis=1)/(iter+1)
            else:
                self.spike_rate[:, iter] = np.sum(spike_rate_window, axis=1)/window_size

            if flag and iter%5000 == 0:
                tempQ = self.Q.numpy()
                tempv = (self.vp-self.vn).numpy()
                maxcut, converged = max_cut(tempQ, tempv)
                print("max cut: %d, #iter: %d" %(maxcut, iter))

            if self.learn:
                self.updateQ()
        
        self.end_time = time.time()
    
    def report_time(self):
        print(f'GTNN run time: {self.end_time-self.start_time}')

    def report_maxcut(self):
        tempQ = self.Q.numpy()
        tempv = (self.vp-self.vn).numpy()
        maxcut, converged = max_cut(tempQ, tempv)
        print("At the current stage, max cut: %d" %(maxcut))
            
    def plot_raster(self):
        global neuron, duration
        spikes_list = list()
        for i in range(neuron):
            spikes_list.insert(i, self.true_spikes[i, np.nonzero(self.true_spikes[i,:])])
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_xlim(0, int(arg_list['TMAX']/arg_list['DT']))
        for i in range(arg_list['NUM_NEURON']):
            ax1.eventplot(spikes_list[i], linelengths=0.8, lineoffsets=i, colors='black')
        # ax1.set_title('Neuron Firing Event')
        ax1.set_xlabel('Time (us)')
        ax1.set_ylabel('Neuron')
        plt.show()
        pass

    def plot_general(self):
        norm = matplotlib.colors.Normalize()
        fig = plt.figure()
        fig.tight_layout(pad=5)
        v_ax = fig.add_subplot(2, 1, 1)
        vnorm_ax = fig.add_subplot(2, 1, 2)

        vnorm = np.linalg.norm((self.vp_ev-self.vn_ev), axis=0)
        vnorm_ax.plot(vnorm.reshape(-1))
        vnorm_ax.set_title('Evolution of Norm')
        vnorm_ax.set_ylabel('Norm of Membrane Potential')
        vnorm_ax.set_xlabel('time(n)')

        v_temp = self.vp_ev.T - self.vn_ev.T
        v_ax.plot(v_temp)
        v_ax.set_title('Evolution of Membrane Potential')
        v_ax.set_ylabel('Membrane Potential (mV)')
        plt.show()
        pass

    def plot_adjacency(self):
        norm = matplotlib.colors.Normalize()
        fig = plt.figure()
        Q_ax = fig.add_subplot(111)

        Q_ax.imshow(self.Q.numpy(), norm=norm, cmap='gray')
        Q_ax.axis('off')
        Q_ax.set_title('Adjacency Matrix')
        plt.show()


