% Load the data
data = load('userdata.mat');  % Load the .mat file
fieldnames(data)  % Check the variable name inside the .mat file
varname = fieldnames(data);  % Get the actual variable name
userdata = data.(varname{1});  % Extract the matrix

% Find non-zero frames
nonzero_frames = squeeze(any(any(userdata, 1), 2));  % Logical index of non-zero frames

% Keep only nonzero frames
userdata_filtered = userdata(:, :, nonzero_frames);

% Save the updated data
save('userdata.mat', 'userdata');
