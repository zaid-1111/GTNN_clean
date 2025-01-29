function GTNN_Headless_Sim(nNeuron, T)
% GTNN_Headless_Sim_Scaled - A scalable version of the GTNN simulation
% Inputs:
%   nNeuron - Number of neurons
%   T       - Total simulation steps

% GTNN_Headless_Sim - A “headless” version of the GTNN demo that
% runs the simulation purely in code, with no GUI or figures.

% ----------------------- USER PARAMETERS ----------------------------


useGPU        = true;     % Set this to 'true' if we want to run on the GPU
plotMembrane  = true;    % If we only want to store results, not plot them
nSpeed        = 1;        % Speed multiplier (like the slider in the GUI)
dt            = 0.001;    % Simulation timestep
tau           = 0.01;     % Time constant
eta           = 0.1;      % Learning rate
learnFlag     = false;     % Whether to enable learning or not
dataflag      = false;    % If true, will use external 'userdata.mat'
repeatdata    = 100;      % # timesteps each data vector is applied
Tiled =       = true;


% If not using external data, specify any DC/AC drive here:
I_input       = zeros(nNeuron,1);   % DC input

%Example DC stimulation on neuron:
I_input(1)    = 0.09;

ac_amp        = zeros(nNeuron,1);   % AC amplitude
freq          = 5*ones(nNeuron,1);  % AC frequency
a             = ones(nNeuron,1);    % "alpha" parameter for each neuron
% --------------------------------------------------------------------

%  If we want to load user data for training:
if dataflag
    load userdata.mat;        % This file must be in the same folder
    [dataLen,dataDim] = size(userdata);
else
    userdata  = []; 
    dataLen = 0; 
    dataDim = 0;
end

% Initialize synaptic weight matrix Q, connectivity mask M, identity I
Q = zeros(nNeuron,nNeuron);
M = ones(nNeuron,nNeuron);
I = eye(nNeuron);

% (Optional) pick any custom Q here.  E.g. uncomment:
 Q = 0.5*randn(nNeuron,nNeuron);
     
% Q(logical(eye(size(Q)))) = 0;  % zero diagonal
Qcustom = [0	-0.25	-0.026	0	0	0
           -0.35	0	-0.14	0	0.13	0
           -0.24	-0.15	0	0.11	0	0
            0	0	0.11	    0	0.434	-0.337
            0	0	0.31	    0.067	0	0.29
            0	0	0	    -0.42	0.068	0];
%Q = Qcustom;
% Make sure everything is on the GPU if desired:
if useGPU
    Q       = gpuArray(Q);
    M       = gpuArray(M);
    I       = gpuArray(I);
    I_input = gpuArray(I_input);
    a       = gpuArray(a);
    ac_amp  = gpuArray(ac_amp);
    freq    = gpuArray(freq);
end

% State variables
vp   = -0.5*ones(nNeuron,1,'like',Q);  % Positive membrane potential branch
vn   = -0.5*ones(nNeuron,1,'like',Q);  % Negative membrane potential branch
Psip = zeros(nNeuron,1,'like',Q);      % +spike flags
Psin = zeros(nNeuron,1,'like',Q);      % -spike flags

% Logging for energy/spikes.  We store the spiking energy in S_av, etc.
win        = 900;           % time window for smoothing
S_hist     = zeros(1,win,'like',Q);
S_av       = zeros(1,T,'like',Q);
spikeEnergy = 0;

% Extra parameters from the GUI version
Lambda = 5;   % threshold parameter
vmax   = 1;   % max membrane potential
vth    = 0;   % threshold for spiking
C      = 1;   % amplitude assigned to Psip / Psin upon spike

% Create variables for user-data iteration
currentIndex  = 1;
currentCount  = 0;
output        = zeros(dataLen,1);  % if we're going to store data usage

% Pre-allocate a buffer for logging membrane potentials (optional).
%  If we only need final Q or final spiking rates, we can skip this.
ylog = zeros(nNeuron, T, 'like', Q);

% --------------- MAIN SIMULATION LOOP ---------------
% We do T steps, each possibly with multiple internal updates (nSpeed).
fprintf('Starting simulation with %d neurons for %d steps...\n',nNeuron,T);
iter = 1;

for t = 1:T
    
    % Repeat "nSpeed" times each ms-step if desired
    for subIter = 1:nSpeed
        
        % Decide on input current (from either data or user‐specified AC/DC)
        if dataflag
            % If using training data from 'userdata'
            if currentCount > repeatdata
                if learnFlag
                    currentIndex = randi(dataLen,1);   % pick random pattern
                else
                    currentIndex = currentIndex + 1;
                    if currentIndex > dataLen
                        currentIndex = 1;
                    end
                end
                currentCount = 0;
            end
            currentCount = currentCount + 1;
            if useGPU
            netI = gpuArray(netI);
            end

            % "userdata" might have fewer dims than nNeuron, so pad
            if dataDim >= nNeuron
                netI = userdata(currentIndex,1:nNeuron).';
            else
                netI = [userdata(currentIndex,:), ...
                        zeros(1,nNeuron - dataDim)].';
            end
            % Optionally add a constant offset:
            netI(end) = 0.1;  % from the old GUI example
        else
            % If not using external data, just AC+DC
            netI = I_input + ac_amp .* sin(2*pi*freq*iter/1000);
        end

        % Convert to GPU if we are using GPU mode:
        

        % -- Core spiking logic from the original code --

        % 1) Check for spiking
        spikedP = (vp > vth);
        spikedN = (vn > vth);
        Psip(spikedP) = C;
        Psin(spikedN) = C;
        vp(spikedP)   = vth;
        vn(spikedN)   = vth;

        % 2) Optionally record the new membrane potential for debugging
        ylog(:,t) = vp;   % or something else (like y in GUI code)
        % 3) Calculate the gradient
        Gp = vp - netI + Q*(vp - vn) + Psip;
        Gn = vn + netI - Q*(vp - vn) + Psin;
        % 4) Update vp, vn
        vp = a.*vp + (dt/tau)*(((vp.^2 - vmax^2).*Gp)./(-vp.*Gp + Lambda*vmax));
        vn = a.*vn + (dt/tau)*(((vn.^2 - vmax^2).*Gn)./(-vn.*Gn + Lambda*vmax));

        % 5) Compute spiking energy
        numSpikes = sum(spikedP) + sum(spikedN);
        S_hist = [S_hist(2:end), numSpikes];   % shift in sliding window
        spikeEnergy = sum(S_hist)/(2*win*nNeuron);        
        S_av(t) = spikeEnergy;

        % 6) Learning
        if learnFlag
            %  Weight update (Psip-Psin)*(vp-vn)'
            Q = Q + 0.5*eta * M .* ((Psip - Psin)*(vp - vn)');
        end

        % 7) Reset spike flags
        Psip(:) = 0;
        Psin(:) = 0;

        % Optional: store some result if dataflag==1 and learnFlag==0
        if dataflag && ~learnFlag
            output(currentIndex) = spikeEnergy;
        end

        iter = iter + 1;        
    end % of sub-iterations

end % of T steps

% Pull back GPU arrays to CPU memory if needed
if useGPU
    Q        = gather(Q);
    ylog     = gather(ylog);
    S_av     = gather(S_av);
end

% Report final connectivity matrix, energy, etc.
fprintf('Simulation complete.\n');
fprintf('Final mean spiking energy = %g.\n', mean(S_av(end-50:end)));

% If we want a quick plot at the end, optionally do it here:
colorMap = repmat(linspace(0,0.7,30)',1,3);
colorMap = [colorMap;[1 1 1];colorMap(end:-1:1,:)];
colorMap(1:30,1) = 1;
colorMap(32:61,3) = 1;
if plotMembrane
    figure;  
    subplot(1,2,1);
    
    imagesc(Q + eye(nNeuron));
    colormap(colorMap)
    
    set(gca,'xtick',1:nNeuron,'ytick',1:nNeuron)
    ylabel('Post-synaptic')
    xlabel('Pre-synaptic')
    title('Connectivity Matrix')
    caxis([-1 1])
    caxis manual
    colorbar


    subplot(1,2,2);
    plot(S_av,'LineWidth',2);
    %set(gca,'YScale','log');
    xlabel('Time step'); ylabel('Spiking Energy');
    grid on; title('Spiking Energy (log scale)');
end

% If needed, save results
% save('GTNN_results.mat','Q','ylog','S_av','-v7.3');
end
