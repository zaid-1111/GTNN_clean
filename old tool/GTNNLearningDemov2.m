% A MATLAB demonstration of GTNN Learning for small scale network 
%
% This function creates an interactive GUI for simulating the learning
% framework in a population of Spiking Growth Transform Neuron Models as  
% described in  Ref. 1. The focus of this GUI is on learning and does not
% provide the user the flexibility to emulate single neuron dynamics
% (bursting, spike rate adaptation,...) as described in Ref. 2. Future
% versions of this GUI will combine both these features.
%
% Before running the script, please ensure that the following MAT files
% are present in the same directory.
% userdata.mat - which stores the matrix userdata. When the User Data Flag
% is enabled, the script uses userdata to generate inputs to the network.
% The script randomly selects one of the vectors and and presents it as
% an input to the network. The parameter Repeat Data - specifies the time
% for which the input is held fixed before randomly selecting another
% vector. 
%  
% The users have the ability to change the input current 
% (DC and AC components) rather than using the User Data. 
% This can be done for
% an individual neuron or for all neurons simultaneously. Clicking reset
% resets all inputs to their default values.
% The GUI can also simulate a network of neurons with inhibitory and
% excitatory connections. There is an option for generating sparse random
% connectome in the GUI. The user can also input their own
% connectome by defining it in the workspace and then importing
% it into the GUI using the Connectivity Matrix Dropdown Menu. The matrix
%  should be a square matrix of size = number of
% neurons. This GUI works best for 1-40 neurons. 
%
% When the learning mode is ON, this toolbox implements spike based
% learning described in the paper (see Ref. 1). The user can observe
% change in network spiking rate as the network learns/memorizes the
% pattern. After learning, the user can save the learned connectome using
% the save button - which can then be retrieved in the next session.
%
% GTNNLearningDemo.m Copyright (c) 2021.
%
% 1. A. Gangopadhyay, S. Chakrabartty, A Sparsity-driven 
% Backpropagation-less Learning Framework using Populations of Spiking 
% Growth Transform Neurons, Frontiers in Neuroscience, 2021.
%
%
% 2. A. Gangopadhyay, D. Mehta, S. Chakrabartty, 
% A Spiking Neuron and Population Model Based on the Growth Transform 
% Dynamical System, Frontiers in Neuroscience, 12 May 2020  
% https://doi.org/10.3389/fnins.2020.00425
% 
% -------------------------------------------------------------------------
% Washington University hereby grants to you a non-transferable,
% non-exclusive, royalty-free, non-commercial, research license to use and
% copy the computer code provided here (the “Software”).  You agree to
% include this license and the above copyright notice in all copies of the
% Software.  The Software may not be distributed, shared, or transferred to
% any third party.  This license does not grant any rights or licenses to
% any other patents, copyrights, or other forms of intellectual property
% owned or controlled by Washington University.  If interested in obtaining
% a commercial license, please contact Washington University's Office of
% Technology Management (otm@dom.wustl.edu).
% 
% YOU AGREE THAT THE SOFTWARE PROVIDED HEREUNDER IS EXPERIMENTAL AND IS
% PROVIDED “AS IS”, WITHOUT ANY WARRANTY OF ANY KIND, EXPRESSED OR IMPLIED,
% INCLUDING WITHOUT LIMITATION WARRANTIES OF MERCHANTABILITY OR FITNESS FOR
% ANY PARTICULAR PURPOSE, OR NON-INFRINGEMENT OF ANY THIRD-PARTY PATENT,
% COPYRIGHT, OR ANY OTHER THIRD-PARTY RIGHT.  IN NO EVENT SHALL THE
% CREATORS OF THE SOFTWARE OR WASHINGTON UNIVERSITY BE LIABLE FOR ANY
% DIRECT, INDIRECT, SPECIAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF OR IN
% ANY WAY CONNECTED WITH THE SOFTWARE, THE USE OF THE SOFTWARE, OR THIS
% AGREEMENT, WHETHER IN BREACH OF CONTRACT, TORT OR OTHERWISE, EVEN IF SUCH
% PARTY IS ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. YOU ALSO AGREE THAT
% THIS SOFTWARE WILL NOT BE USED FOR CLINICAL PURPOSES.
% -------------------------------------------------------------------------


function GTNNLearningDemo
% function GT_NeuronModel

% First get the input number of neurons
prompt={'Enter number of neurons between 1 and 40'};
name = 'Number of neurons';
defaultans = {'5'};
answer = inputdlg(prompt,name,[1 40],defaultans);
temp = str2double(answer);
if isempty(answer)
    disp("Error in fetching number of neurons\n");
    return
elseif ~isnan(temp) && temp>0 && temp<41
    nNeuron = round(temp);
else
    disp('Number of neurons needs to be integer between 1 and 100')
    return
end
% Parameters
% Start from a random or fixed initial point


% Set synaptic weight matrix
Q = zeros(nNeuron,nNeuron);

% Mask matrix - which shows which neurons are phyically connected
M = ones(nNeuron,nNeuron);
I = eye(nNeuron,nNeuron);
I_input = 0*ones(nNeuron,1);
a = ones(nNeuron,1);


nSpeed = 1;
L = 1000;
repeatdata = 100;
ac_amp = zeros(nNeuron,1);
freq = 5*ones(nNeuron,1);
y = zeros(nNeuron,L);
I_hist = zeros(nNeuron,L);
userdata = [];
load userdata;
output = [];

nSelect = 1;
nDisp = 1;
pauseFlag = 0;
learnFlag = 0;
datalen = 0;
datadim = 0;
dataflag = 0;
nStr = cell(1,nNeuron+1);
for n1 = 1:nNeuron
    nStr{n1} = num2str(n1);
end
nStr{nNeuron+1}='All';
% Figure
figNumber = figure(1);
clf;
set(figNumber,'NumberTitle','off',...
    'Name','Growth Transform Neuron Model',...
    'Units','normalized','toolbar','figure',...
    'Position',[0.05 0.1 0.9 0.8]);

% Control buttons

% Input fields
nInputFields = 7;
ycoord = linspace(1,0,nInputFields+1);
yHt = 1/nInputFields - 0.05;
pnl_input = uipanel('Title','Input Fields','FontSize',10,...
    'BackgroundColor','white',...
    'Position',[.31 .1 .35 .3]);
inputFieldStrings = {'Input Stimulus (DC current) = ','Input Stimulus (AC Amplitude) = ','Input Stimulus (AC Frequency) = ','User Data Repeatition = ','Inter-spike Interval: ', 'User Data Flag: '};
inputFieldDefaults = {'0','0','5','100','1',''};

inFieldDisp = cell(1,6);
for iter = 1:numel(inFieldDisp)
    uicontrol('Parent',pnl_input,'Style','text', 'Units','normalized', ...
        'BackgroundColor','white','Position',[0.1 ycoord(iter+2)+0.03 0.4 yHt],'string',inputFieldStrings{iter},...
        'FontSize',9, 'HorizontalAlignment', 'right');
    
    inFieldDisp{iter} = uicontrol('Parent',pnl_input,'Style','text', 'Units','normalized', ...
        'BackgroundColor','white','Position',[0.51 ycoord(iter+2)+0.03 0.1 yHt],'string',inputFieldDefaults{iter},...
        'FontSize',9, 'HorizontalAlignment', 'left');
end
inField{1} = uicontrol('Parent',pnl_input,'Style','slider',...
    'Min',-0.9,'Max',0.9, 'SliderStep',[0.05 0.2], 'Units','normalized', ...
    'Position',[0.62 ycoord(3)+0.04 0.3 yHt-0.01],...
    'tag','input','Callback',@changepars);


inField{2} = uicontrol('Parent',pnl_input,'Style','slider',...
    'Min',0,'Max',0.2, 'SliderStep',[0.05 0.2],'Units','normalized', ...
    'Position',[0.62 ycoord(4)+0.06 0.3 yHt-0.01],...
    'Value',ac_amp(1),'tag','ac_amp','Callback',@changepars);


inField{3} = uicontrol('Parent',pnl_input,'Style','slider',...
    'Min',0,'Max',10, 'SliderStep',[0.1 0.2],'Units','normalized',...
    'Position',[0.62 ycoord(5)+0.04 0.3 yHt-0.01],...
    'Value',freq(1),'tag','freq','Callback',@changepars);


inField{4} = uicontrol('Parent',pnl_input,'Style','slider',...
    'Min',1,'Max',1000, 'SliderStep',[0.01 0.1], 'Units','normalized', ...
    'Position',[0.62 ycoord(6)+0.04 0.3 yHt-0.01],...
    'Value',repeatdata,'tag','repeat','Callback',@changepars);


inField{5} = uicontrol('Parent',pnl_input,'Style','slider',...
    'Min',0,'Max',1, 'SliderStep',[0.05 0.2], 'Units','normalized', ...
    'Position',[0.62 ycoord(7)+0.04 0.3 yHt-0.01],...
    'Value',a(1),'tag','alpha','Callback',@changepars);

inField{6} = uicontrol('Parent',pnl_input,'Style','checkbox',...
    'Min',0,'Max',1, 'Units','normalized', ...
    'Position',[0.51 ycoord(8)+0.04 0.05 yHt-0.01],...
    'BackgroundColor','white','tag','Data','Callback',@changepars);

uicontrol('Parent',pnl_input, 'Units','normalized', ...
        'Position',[0.62 ycoord(8)+0.02 0.1 yHt+0.01],...
        'string','save','tag','save','Callback',@changepars);


if nNeuron>1
    uicontrol('Parent',pnl_input,'Style','text', 'Units','normalized', ...
        'BackgroundColor','white','Position',[0.2 ycoord(2)+0.03 0.3 yHt],'string','Select neuron',...
        'HorizontalAlignment', 'right','FontSize',9);
    uicontrol('Parent',pnl_input,'Style', 'popup', 'Units','normalized',...
        'String', nStr,...
        'Position', [0.51 ycoord(2)+0.035 0.2 yHt],...
        'tag','neurons','Callback', @changepars);
end


% Simulation speed
uicontrol('Style','text', 'Units','normalized', ...
    'Position',[0.7  0.96 0.2 0.03],'string','Simulation speed','FontSize',12);
uicontrol('Style', 'slider',...
    'Min',1,'Max',10,'Value',nSpeed, 'Units','normalized', ...
    'Position',[0.7  0.92 0.2 0.03],'SliderStep',[0.1 0.2],...
    'tag','speed','Callback',@changepars);

% Pause button
uicontrol('Style', 'togglebutton','String','Pause/Resume',...
    'Min',0,'Max',1,'Value',0, 'Units','normalized', ...
    'Position',[0.8 0.02 0.1 0.05],...
    'tag','pauseflag','Callback',@changepars);

% Reset button
uicontrol('Style', 'togglebutton','String','Reset',...
    'Min',0,'Max',1,'Value',0, 'Units','normalized', ...
    'Position',[0.65 0.02 0.1 0.05],...
    'tag','reset','Callback',@changepars);

% Learn button
uicontrol('Style', 'togglebutton','String','Learn',...
    'Min',0,'Max',1,'Value',0, 'Units','normalized', ...
    'Position',[0.50 0.02 0.1 0.05],...
    'tag','learnflag','Callback',@changepars);

h1 = axes('Position',[0.05 0.5 0.9 0.4]);
hold on

nAx = cell(1,nNeuron);
for n1 =1:nNeuron
    nAx{n1} = plot(h1,1:L,y(n1,:)+n1,'LineWidth',2);
end



axis manual
axis([0 1000 0 nNeuron+1])
title('Membrane potential')
xlabel('Time (ms)');
ylabel('Neuron Index');
set(gca,'ytick',1:nNeuron)

h2 = axes('Position',[0.7 0.15 0.25 0.25]);
S_av = zeros(1,1000);
I_Ax = plot(h2,1:1000,S_av,'b','LineWidth',2);

axis manual
axis([0 1000 0 0.25/nNeuron])
title('Spiking Energy')
xlabel('Time (ms)');
ylabel('Energy (a.u.)');
grid on;

colorMap = repmat(linspace(0,0.7,30)',1,3);
colorMap = [colorMap;[1 1 1];colorMap(end:-1:1,:)];
colorMap(1:30,1) = 1;
colorMap(32:61,3) = 1;
if nNeuron>1
    h3=axes('Position',[0.05 0.1 0.25 0.25]);
    conn_im = imagesc(h3,Q+I);
    
    colormap(colorMap)
    set(gca,'xtick',1:nNeuron,'ytick',1:nNeuron)
    ylabel('Post-synaptic')
    xlabel('Pre-synaptic')
    title('Connectivity Matrix')
    uicontrol('Style', 'popup', 'Units','normalized',...
        'String', {'Identity','Random Sparse','Random Non-sparse','Random PD','Saved'},...
        'Position', [0.08  0.22 0.2 0.2],...
        'tag','Q_mat','Callback', @changeQ);
    caxis([-1 1])
    caxis manual
    colorbar    
end




% Complementary variables and their initial condition
vp = -0.5*ones(nNeuron,1);
Psip = zeros(nNeuron,1);
vn = -0.5*ones(nNeuron,1);
Psin = zeros(nNeuron,1);


% Hyperparameters
C = 1; % Spike rate
Lambda = 5; % Lambda
vmax = 1; % Maximum membrane potential
vth = 0;


% Simulation parameters
dt = 0.001; % Sampling time in seconds
T = 500; % Total simulation time in seconds
tau = 0.01; % Time-constant
eta = 0.1; % Learning rate
iter = 1; % simulation counter
Titer = 100; % Duration to perform forward and backward pass.

currind = 1;
currentcount = 0;
win = 900;
S_hist = zeros(1,win);
currind = 1;
spikeenergy = 0;

while ishandle(figNumber)
    for c1 = 1:nSpeed        
        
        % If user data flag is selected
        if dataflag > 0,
            % Pick a random training data and repeat for repeatdata
            if (currentcount > repeatdata)
               if learnFlag > 0
                  currind = randi(datalen,1);
               else
                  currind = currind + 1;
                  output(currind) = spikeenergy;
                  if (currind > datalen)
                      currind = 1;
                  end
                  
                  % Reset all the membrane potentials
                  %vp = -0.5*ones(nNeuron,1);
                  %Psip = zeros(nNeuron,1);
                  %vn = -0.5*ones(nNeuron,1);
                  %Psin = zeros(nNeuron,1);
               end
               currentcount = 0;
            end
            if datadim >= nNeuron-1,
               % data dimension is greater than the number of neurons
               % then choose only datadim inputs
               netI = [userdata(currind,1:nNeuron) 0.1]';
               %netI = [0.1 userdata(currind,2:nNeuron) userdata(currind,1)]';              
            else
               % if data dimension is smaller than number of neurons
               % then pad with zeros
               netI = [userdata(currind,:) zeros(1,nNeuron-1-datadim) 0.1]';
               %netI = [0.1 userdata(currind,2:datadim) zeros(1,nNeuron-1-datadim) userdata(currind,1)]';
            end
            currentcount = currentcount + 1;
        else
            % External stimuli current - variable b
            netI = I_input+ac_amp.*sin(2*pi*freq*iter/1000); % Net differential input current
        end;
        

        % ind is the flag that indicates which neurons should fire
        % and then reset the membrane potential to threshold thr
        % Find which of the neurons spiked
        indp = find(vp > vth);
        indn = find(vn > vth);
        Psip(indp) = C;
        Psin(indn) = C;
        vp(indp) = vth;
        vn(indn) = vth;
        
         % Create the membrane potential buffer
        y = [y(:,2:end), (vp+ 0.5*Psip)];        
    
        % Calculate the gradient
        Gp = vp - netI + Q*(vp - vn) + 1*Psip;
        Gn = vn + netI - Q*(vp - vn) + 1*Psin;
    
        % Now update the membrane potential   
        vp = a.*vp + (dt/tau).*(((vp.^2 - vmax^2).*Gp)./(-vp.*Gp + Lambda*vmax));
        vn = a.*vn + (dt/tau).*(((vn.^2 - vmax^2).*Gn)./(-vn.*Gn + Lambda*vmax));
    
        
        % Energy due to spiking
        Sn = sum((vp > 0)) + sum((vn > 0));
        S_hist = [S_hist(2:end), Sn];
        spikeenergy = sum(S_hist)/(2*win*nNeuron);
        S_av = [S_av(2:end), spikeenergy];        
        
        if learnFlag > 0,
            Q = Q + 0.5*eta*M.*((Psip - Psin)*(vp-vn)');  
%            Q = Q - diag(diag(Q),0);              
        end
               
        % Reset the spike
        Psip = zeros(nNeuron,1);
        Psin = zeros(nNeuron,1);
%         vp(indp) = -0.2;
%         vn(indn) = -0.2;

        iter = mod(iter,100000)+1;
    end
    for n1 =1:nNeuron
        set(nAx{n1},'ydata',y(n1,:)+n1) % Update the membrane potential plot
    end
    
    set(I_Ax,'ydata',S_av) % Update the input current plot   
    drawnow

    set(conn_im,'cdata',Q+I);
    drawnow

    
    while pauseFlag && ishandle(figNumber)
        drawnow
        pause(0.1)
    end
end
% Functions
    function changepars(source, ~)
        t = source.Tag;
        switch t
            case 'input'
                nv=source.Value;
                if isempty(nv)
                    nv=I_input(nSelect,1);
                end
                I_input(nSelect,1)=nv;
                set(inFieldDisp{1},'string',num2str(nv,'%.2f'));
            case 'speed'
                nv=round(source.Value);
                if isempty(nv)
                    nv=nSpeed;
                end
                nSpeed=nv;
                set(source,'string',num2str(nv));
            case 'ac_amp'
                nv=source.Value;
                if isempty(nv)
                    nv=ac_amp(nSelect,1);
                end
                ac_amp(nSelect,1)=nv;
                set(inFieldDisp{2},'string',num2str(nv,'%.2f'));
            case 'repeat'
                nv=source.Value;
                repeatdata = nv;
                set(inFieldDisp{4},'string',num2str(nv,'%2d'));
            case 'freq'
                nv=source.Value;
                if isempty(nv)
                    nv=freq(nSelect,1);
                end
                freq(nSelect,1)=nv;
                set(inFieldDisp{3},'string',num2str(nv,'%.2f'));
            case 'alpha'
                nv=source.Value;
                if isempty(nv)
                    nv=a(nSelect,1);
                end
                a(nSelect,:)=nv;
            case 'Data'
                nv=source.Value;
                dataflag = nv;
                load userdata;
                [datalen,datadim] = size(userdata);
                output = zeros(datalen,1);
               
            case 'save'
                save Qprev Q output;
            case 'neurons'
                val = source.Value;
                if val<=nNeuron
                    nSelect = val;
                    nDisp = val;
                else
                    nSelect = 1:nNeuron;
                end
            case 'pauseflag'
                pauseFlag = source.Value;
            case 'learnflag'
                learnFlag = source.Value;                
            case 'reset'
                I_input(:,1) = 0;
                ac_amp(:,1) = 0;
                freq(:,1) = 5;
                dataflag = 0;
                vp = -0.5*ones(nNeuron,1);
                Psip = zeros(nNeuron,1);
                vn = -0.5*ones(nNeuron,1);
                Psin = zeros(nNeuron,1);
        end
        displayParams;
    end

    function displayParams        
        set(inFieldDisp{1},'string',num2str(I_input(nDisp)));
        set(inFieldDisp{2},'string',num2str(ac_amp(nDisp)));
        set(inFieldDisp{3},'string',num2str(freq(nDisp)));
        set(inFieldDisp{4},'string',num2str(repeatdata));
        set(inFieldDisp{5},'string',num2str(a(nDisp)));        
        set(inField{1},'Value',I_input(nDisp));
        set(inField{2},'Value',ac_amp(nDisp));
        set(inField{3},'Value',freq(nDisp));
        set(inField{4},'Value',repeatdata);
        set(inField{5},'Value',a(nDisp));
        set(inField{6},'Value',dataflag);
    end
    function changeQ(source,~)
        val = source.Value;
        switch val
            case 1
                Q = eye(nNeuron, nNeuron);
            case 2
                Q = full(0.1*sprandn(nNeuron, nNeuron,0.1));
                for nf = 1:nNeuron
                    Q(nf,nf) = 0;
                end
                M(abs(Q) < 0.00001) = 0;
            case 3
                Q = 1*(rand(nNeuron, nNeuron)-0.5);
                for nf = 1:nNeuron
                    Q(nf,nf) = 0;
                end
                M(abs(Q) < 0.00001) = 0;                
            case 4
                Q = 1*(rand(nNeuron, nNeuron)-0.5);
                Q = Q*Q';
                for nf = 1:nNeuron
                    Q(nf,nf) = 0;
                end 
                M(abs(Q) < 0.00001) = 0;                
            case 5
                %nv = evalin('base','Q');
                load Qprev;
                nv = Q;
                disp(nv)
                if size(nv,1) == size(nv,2) && size(nv,2) == nNeuron
                    Q = nv;
                else
                    disp('Q is not a square matrix of size = number of neurons')
                    disp(size(nv,1))
                    disp(size(nv,2))
                end
                for nf = 1:nNeuron
                    Q(nf,nf) = 0;
                end
                M(abs(Q) < 0.00001) = 0;                
        end
        conn_im.CData = Q+I;
    end
end