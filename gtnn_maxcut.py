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
import numpy as np
import matplotlib.pylab as plt
import matplotlib
from gtnn_config import arg_list
from gtnn_object import GTNN

neuron = 800

arg_list['NUM_NEURON'] = neuron
arg_list['QFILE'] = './data/graphs/G15.txt'
arg_list['VMAX'] = 1
arg_list['DT'] = 0.001
arg_list['TMAX'] = 10
arg_list['TAU'] = 0.01
arg_list['ETA'] = 0.002
arg_list['LAMBDA'] = 10
arg_list['C'] = 1
arg_list['VTH'] = 0

# Run in max cut mode
en_maxcut = 1
# GTNN Initialization
myGTNN = GTNN()
myGTNN.init_v('None')
myGTNN.init_Q('user data')
myGTNN.init_mask()
# Plot the Q/adjacency matrix
myGTNN.plot_adjacency()
# No Q adaptation
myGTNN.learn = 0
# Zero inputs
b_temp = np.zeros((neuron, 1))
myGTNN.update(b = b_temp.reshape(neuron,1))
# Sanity check before run
myGTNN.dimension_check()
myGTNN.run(en_maxcut)
myGTNN.report_maxcut()
myGTNN.report_time()
myGTNN.plot_general()

# Continue running
# Update time and input before calling update()
arg_list['TMAX'] = 5
b_temp = np.zeros((neuron, 1))
myGTNN.update(b = b_temp.reshape(neuron,1))
# Refresh the recording of membrane potential
myGTNN.refresh_record()
myGTNN.dimension_check()
myGTNN.run(en_maxcut)
myGTNN.report_maxcut()
myGTNN.report_time()
myGTNN.plot_general()