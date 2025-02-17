function GTNN_Headless_Sim()
    % GTNN_Headless_Sim - Reads simulation parameters from a JSON config file
    % and runs the simulation with those settings.

    % 1) Read the JSON config file
    fid = fopen("configFile.json");
    raw = fread(fid, inf);
    fclose(fid);
    jsonText = char(raw');
    cfg = jsondecode(jsonText);  % Convert JSON to MATLAB struct
    
    % 2) Extract required parameters
    nNeuron        = cfg.nNeuron;
    T              = cfg.T;
    useGPU         = cfg.useGPU;
    plotMembrane   = cfg.plotMembrane;
    nSpeed         = cfg.nSpeed;
    dt             = cfg.dt;
    tau            = cfg.tau;
    eta            = cfg.eta;
    learnFlag      = cfg.learnFlag;
    dataflag       = cfg.dataflag;
    repeatdata     = cfg.repeatdata;
    Tiled          = cfg.Tiled;
    TileDivisor    = cfg.TileDivisor;
    sparsityFactor = cfg.sparsityFactor;
    Lambda         = cfg.Lambda;
    vmax           = cfg.vmax;
    vth            = cfg.vth;
    C              = cfg.C;
    feedForward    = cfg.feedForward;
    feedBack       = cfg.feedBack;


% DC/AC input stimuli
I_input  = zeros(nNeuron,1);
I_input(1) = 0.09;           % small DC bias on neuron #1
ac_amp   = zeros(nNeuron,1); % AC amplitude
freq     = 5*ones(nNeuron,1);% AC frequency
a        = ones(nNeuron,1);  % alpha parameter

%% ------------------- Load user data if needed -----------------------
if dataflag
    load userdata.mat;        
    [dataLen,dataDim] = size(userdata);
else
    userdata  = []; 
    dataLen   = 0; 
    dataDim   = 0;
end

%% ------------------- Initialize Q, M, Identity ----------------------
% Q: the synaptic weight matrix (will be sparse).
TileSize = nNeuron/TileDivisor;
Q = 0.5 * randn(nNeuron,nNeuron);

% Optionally tile Q by setting only certain diagonal blocks:
if Tiled
    tileSize = nNeuron / TileDivisor;
    Q_tiled  = sparse(nNeuron,nNeuron);
    
    % 1) Copy the main diagonal sub-blocks into Q_tiled:
    for iBlock = 1:TileDivisor
        idx = (iBlock-1)*tileSize + (1:tileSize);
        Q_tiled(idx, idx) = Q(idx, idx);
    end
    
    % 2) Optionally add feed-forward sub-blocks (above diagonal tiles)
    if feedForward
        % For each diagonal tile, we place some connections in the next tile up:
        for iBlock = 1:(TileDivisor - 1)
            rowIdx = (iBlock-1)*tileSize + (1:tileSize);
            colIdx = (iBlock)*tileSize   + (1:tileSize);
            
            % For example, keep ~10% of these weights at random:
            mask = (rand(tileSize) < sparsityFactor);
            Q_tiled(rowIdx, colIdx) = Q(rowIdx, colIdx) .* mask;
        end
    end
    
    % 3) Optionally add feed-back sub-blocks (below diagonal tiles)
    if feedBack
        % For each diagonal tile, we place some connections in the previous tile down:
        for iBlock = 2:TileDivisor
            rowIdx = (iBlock-1)*tileSize + (1:tileSize);
            colIdx = (iBlock-2)*tileSize + (1:tileSize);
            
            mask = (rand(tileSize) < sparsityFactor);
            Q_tiled(rowIdx, colIdx) = Q(rowIdx, colIdx) .* mask;
        end
    end
    
    % Finally replace Q with Q_tiled:
    Q = Q_tiled;
else
    % Non-tiled case (if Tiled = false), just do Q = sparse(Q)
    Q = sparse(Q);
end
%% Add some Selected feed-forward connections
% Q(TileSize-2,TileSize+3) = 0.56;
% Q(TileSize-4,TileSize+6) = 0.78;
% Q(TileSize-3,TileSize+5) = -0.64;


 
M = double(Q ~= 0);  % M(i,j) = 1 where Q(i,j)~=0, 0 otherwise This makes only initial connections trainable

I = eye(nNeuron);

%% ---------------- Move to GPU if desired ---------------------------
% We can keep Q as a sparse GPU array.  For all vectors that need
%   elementwise ">" or other ops, we use dense GPU arrays.
colorMap = repmat(linspace(0,0.7,30)',1,3);
colorMap = [colorMap;[1 1 1];colorMap(end:-1:1,:)];
colorMap(1:30,1) = 1;
colorMap(32:61,3) = 1;
if plotMembrane
    figure;
    
    imagesc(Q + eye(nNeuron));
    colormap(colorMap)    
    %set(gca,'xtick',1:nNeuron,'ytick',1:nNeuron)
    ylabel('Post-synaptic')
    xlabel('Pre-synaptic')
    title('Initial Connectivity Matrix')
    caxis([-1 1])
    caxis manual
    colorbar
end
if useGPU
    Q = gpuArray(Q);      % sparse GPU array
    M = gpuArray(M);     
    I = gpuArray(I);      % dense identity
    I_input = gpuArray(I_input); 
    a = gpuArray(a);
    ac_amp = gpuArray(ac_amp);
    freq = gpuArray(freq);
end

%% ---------------- State variables (dense GPU arrays) ---------------
% DO NOT do 'like', Q because Q is sparse.  We'll create them as dense:
if useGPU
    vp   = gpuArray(-0.5*ones(nNeuron,1));  % positive membrane potential
    vn   = gpuArray(-0.5*ones(nNeuron,1));  % negative membrane potential
    Psip = gpuArray(zeros(nNeuron,1));      % +spike flags
    Psin = gpuArray(zeros(nNeuron,1));      % -spike flags
else
    vp   = -0.5*ones(nNeuron,1);
    vn   = -0.5*ones(nNeuron,1);
    Psip = zeros(nNeuron,1);
    Psin = zeros(nNeuron,1);
end

%% ----------------- Logging for energy/spikes ------------------------
win          = 900; 
S_hist       = zeros(1,win,'like',vp);   % 'like', vp => dense
S_av         = zeros(1,T,'like',vp);
spikeEnergy  = 0;

%% ----------------- Hyperparameters for GTNN ------------------------

%% ----------------- (Optional) for user-data iteration --------------
currentIndex  = 1;
currentCount  = 0;
output        = zeros(dataLen,1,'like',vp);

%% ----------------- Pre-allocate ylog for debugging ------------------
ylog = zeros(nNeuron,T,'like',vp);

%% ----------------- MAIN SIMULATION LOOP -----------------------------
fprintf('Starting simulation with %d neurons for %d steps...\n', nNeuron, T);
iter = 1;

for t = 1:T
    
    % Repeat "nSpeed" times each ms-step if desired
    for subIter = 1:nSpeed
        
        %% 1) Compute net input current
        if dataflag
            % If using training data from 'userdata'
            if currentCount > repeatdata
                if learnFlag
                    currentIndex = randi(dataLen,1);
                else
                    currentIndex = currentIndex + 1;
                    if currentIndex > dataLen
                        currentIndex = 1;
                    end
                end
                currentCount = 0;
            end
            currentCount = currentCount + 1;
            
            if dataDim >= nNeuron
                netI = userdata(currentIndex,1:nNeuron).';
            else
                netI = [userdata(currentIndex,:), zeros(1,nNeuron - dataDim)].';
            end
            netI(end) = 0.1;  
            
            if useGPU
                netI = gpuArray(netI);
            end
        else
            % AC+DC
            netI = I_input + ac_amp .* sin(2*pi*freq*iter/1000);
        end

        %% 2) Check for spiking
        spikedP = (vp > vth);
        spikedN = (vn > vth);
        Psip(spikedP) = C;
        Psin(spikedN) = C;
        vp(spikedP)   = vth;
        vn(spikedN)   = vth;

        %% 3) Optionally record the new membrane potential
        ylog(:,t) = vp; 
        
        %% 4) Calculate the gradient
        %   Q*(vp - vn) is valid if Q is sparse (GPU) and (vp - vn) is dense (GPU).
        Gp = vp - netI + Q*(vp - vn) + Psip;
        Gn = vn + netI - Q*(vp - vn) + Psin;
        
        %% 5) Update vp, vn
        vp = a.*vp + (dt/tau)*(((vp.^2 - vmax^2).*Gp)./(-vp.*Gp + Lambda*vmax));
        vn = a.*vn + (dt/tau)*(((vn.^2 - vmax^2).*Gn)./(-vn.*Gn + Lambda*vmax));

        %% 6) Spiking energy
        numSpikes    = sum(spikedP) + sum(spikedN);
        S_hist       = [S_hist(2:end), numSpikes];
        spikeEnergy  = sum(S_hist)/(2*win*nNeuron);        
        S_av(t)      = spikeEnergy;

        %% 7) Learning
        if learnFlag
            Q = Q + 0.5*eta * M .* ((Psip - Psin)*(vp - vn)');
        end

        %% 8) Reset spike flags
        Psip(:) = 0;
        Psin(:) = 0;

        %% If using data but not learning, store energy
        if dataflag && ~learnFlag
            output(currentIndex) = spikeEnergy;
        end
        
        iter = iter + 1;        
    end
end

%% ----------------- Pull back to CPU if needed -----------------------
if useGPU
    Q    = gather(Q);
    ylog = gather(ylog);
    S_av = gather(S_av);
end

%% ----------------- Final Stats & Optional Plot ----------------------
fprintf('Simulation complete.\n');
fprintf('Final mean spiking energy = %g.\n', mean(S_av(end-50:end)));

if plotMembrane
    figure;  
    subplot(1,2,1);
    
    imagesc(Q + eye(nNeuron));
    colormap(colorMap)
    
    %set(gca,'xtick',1:nNeuron,'ytick',1:nNeuron)
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
    grid on; title('Spiking Energy');
end

end
