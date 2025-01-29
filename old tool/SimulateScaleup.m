% ScaleSimulationAndPlot.m
% Script to measure simulation time for different numbers of neurons

% List of neuron counts to simulate
neuronCounts = [1000,10000,15000,20000,21000,22000,23000];
timeTaken = zeros(size(neuronCounts));  % Initialize array to store times

% Simulation parameters
T = 1000;  % Total time steps for each simulation

for i = 1:length(neuronCounts)
    nNeuron = neuronCounts(i);
    fprintf('Simulating %d neurons...\n', nNeuron);
    
    % Start timing
    tic;
    
    % Call the simulation function with the current neuron count
    GTNN_Headless_Sim(nNeuron, T); 
    
    % End timing
    timeTaken(i) = toc;
    
    fprintf('Simulation with %d neurons completed in %.2f seconds.\n', ...
            nNeuron, timeTaken(i));
end

% Plot the results
figure;
plot(neuronCounts, timeTaken, '-o', 'LineWidth', 2);
xlabel('Number of Neurons');
ylabel('Time Taken (seconds)');
title('Simulation Time vs Number of Neurons');
grid on;
%set(gca, 'XScale', 'YScale');
