nNeuron = 64; % Example, must be divisible by 8
Q = 0.5*randn(nNeuron, nNeuron);

% Determine block size
blockSize = nNeuron / 8;

% Initialize a sparse zero matrix
Q_blocked = sparse(nNeuron, nNeuron);

% Fill diagonal blocks
for i = 1:8
    idx = (i-1)*blockSize + (1:blockSize); % Indices for the block
    Q_blocked(idx, idx) = Q(idx, idx); % Assign diagonal blocks
end

% Display the sparsity pattern
spy(Q_blocked);
