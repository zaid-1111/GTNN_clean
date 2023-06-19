# GTNN_Tool
## Introduction
GTNN_tool is an open source GUI program that implements large-scale growth transform neural network in real-time. GTNN_tool provides full configurability of the neural network for the users, including the dynamics of the neural membrane potential evolution, inputs to the neurons, and connectivity among neurons. The tool also contains visuliazed probes that monitor the real-time power consumption of the network and the membrane potential updates of user-specified neurons. GTNN_tool grants user accessibity of large-scale GT neural network, which can be configured to take on different optimization tasks.  
## First Release
The first release of this program aims to demonstrate the network dynamics of the GTNN in a larger scale than the previous platforms. Multiple pre-defined modes aiming to solve different optimization problem will be added in future releases.  
## Installation
Run installation script **install.sh** in the cloned repository  
`$ ./install.sh`  
After the required packages have been installed, then launch the GUI prgram by  
`$ ./gtnn_gui.py`
## GUI Interface
![GUI](/figures/fig_gui.png)
## Membrane Potential Probe
The probes provide real-time visualization of membrane potential updates for up to 5 different neurons. The probes take on indices from the user and display the real-time membrane potential update of the neurons specified by the incdices to the left of the membrane potential plot.  
## Energy Plot
This real-time energy plot monitors the power consumption of the entire network in arb. unit.
## Neural Model Parameters
User can configure the dynamics of the individual neuron through this panel. In the first release, the MODE dropbox contains 3 pre-defined mode, normal growth transform update rule and tweaked update rule for routing problem and combinatorial problems respectively, and the number of neurons should be kept below 100,000. RIGHT NOW ONLY NORMAL MODE WAS IMPLEMENTED.
## Input Parameters
The GTNN_tool currently supports three modes of inputs: random, user file, and zero. Under random mode, one random input is generated for each neuron. The user can also specify a file that contains a N-by-1 array that represents inputs for each of the N neurons. 
## Neural Network Connectivity Parameters
5 modes have been implemented to generate different neural connectivity graphs: user identity, user feedforward, random identity, random feedforward, user data.  
  
User identity and user feedforward modes allow user to specify certain constraints on the topology of the nuronal connectivity matrix. Thess modes take 3 parameter: recip format file, layer format file, and btwn layer overlap %
 - **inter-layer format file** This parameter reads a user defined file with the format (data, row, col). Eash of the data, row, col, is a column vector with size equal to the number of inter-layer connections of the neural network. Data denotes the strength of the synaptic connection, row and col together encode the coordinate of that synaptic connection in the connectivity matrix. Each element of data should have an absolute value smaller than 1, each of the elements in row and col should be integer and smaller than the number of neuron of the network.  
 - **layer format file** This parameter reads a user defined file with the format (layer_sizes). Layer_sizes should be either a vector of size equal to the number of layers of the connectivity matrix. The sum of the elements in layer_sizes should equal to the number of neurons of the network.  
 - **btwn layer overlap** 
 This parameter determines the percentage of overlap between layers in the network.
  
  
Random identity and random feedforward modes generate random connectivity matrix under specified parameters. These modes take three parameters: overall density %, #inter-layer connections, and btwn layer overlap %.  
 - **overall density value** This parameter dictates the number of layers (1/overall density) of the neural network. It is recommended to have density less than 0.01% for networks that have more than 50,000 neurons.  
 - **number of inter-layer connections** This parameter determines the number of inter-layer connections of the neural network. The inter-layer connections will be evenly generated across the entire network.  
 - **btwn layer overlap** This parameter determines the percentage of overlap between layers in the network.
   


User data mode takes 1 parameter that reads the entire connectivity matrix of the neural network in (data, row, col) format with constraints similar to user identity and user feedforward mode.


