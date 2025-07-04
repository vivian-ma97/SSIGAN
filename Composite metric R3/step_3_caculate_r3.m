% Process all lakes to find the global reference point (closest red point to origin)
% Then calculate distances from all other points to this reference point

clc;
clear;

% All data files
filelist1 = dir('Nomalized_Anomaly_score\*.mat');  %Nomalized anomaly score path
filelist2 = dir('final_CBRP_chu\*.mat');   %% Cbrp result path
filelist3 = dir('anomaly_CBRP_chu\*.mat');
len_file = length(filelist1);
max_CBRP = 90.1552;

% Output folder
save_folder = 'fig_output_distance_kenny';
if ~exist(save_folder, 'dir')
    mkdir(save_folder);
end

% Initialize containers for all red points (to find the global reference)
all_red_x = [];      % Red points x coordinates
all_red_y = [];      % Red points y coordinates

% First pass: Collect all red points to find the global reference
for i = 1:len_file
    file_name1 = filelist1(i).name;
    
    % Load anomaly_CBRP data (contains red points)
    AC_path = fullfile('anomaly_CBRP_chu\', file_name1);    %%For the inventoried subglacial lakes in the AGAP region, the combined Anomaly_score and CBRP metrics have been organized and stored in the folder named Acope.
    load(AC_path);      % X_anomaly, Y_CBRP
    
    % Calculate normalized coordinates for red points
    x_red = X_anomaly(:);

    y_red =  1 - Y_CBRP(:) ./ max_CBRP;

    % Accumulate red points
    all_red_x = [all_red_x; x_red];
    all_red_y = [all_red_y; y_red];
end

% Find the red point closest to origin (0,0)
red_distances = sqrt(all_red_x.^2 + all_red_y.^2);
[~, idx_global_min] = min(red_distances);
x_ref = all_red_x(idx_global_min);
y_ref = all_red_y(idx_global_min);




all_r1 =[];
all_r2 = [];
all_r3 = [];
all_r3_2 = [];
fprintf('Global reference point found at (%.4f, %.4f)\n', x_ref, y_ref);

% Second pass: Process each file and calculate distances to reference point
for i = 1:len_file
    disp(i)
    file_name1 = filelist1(i).name;
    fprintf('Processing file %d/%d: %s\n', i, len_file, file_name1);
    
    % Construct paths
    full_path = fullfile('Nomalized_Anomaly_score\', file_name1);
    CBRP_path = fullfile('final_CBRP_chu\', file_name1);
    
    % Load data
    load(full_path);    % anomaly_scores
    load(CBRP_path);    % P_Bed

    
    clear x_points;
    clear y_points;
    % Calculate normalized coordinates
    x_points = anomaly_scores(:);
    y_points = 1 - P_Bed(:)./max_CBRP;
    
    % Calculate distances to reference point
    distances = sqrt((x_points - x_ref).^2 + (y_points - y_ref).^2);
    disp(max(y_points));
    disp(min(y_points));
    


    
    % Example: Save distances back to a file
    [~, basename, ~] = fileparts(file_name1);
    save(fullfile(save_folder, [basename '.mat']), 'distances', "distances_2");
    all_r1 = [all_r1, x_points'];
    all_r2 = [all_r2, y_points'];
    all_r3 = [all_r3, distances'];

end

disp('All files processed.');
