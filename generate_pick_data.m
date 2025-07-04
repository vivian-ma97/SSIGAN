clc;
clear;

filelist1 = dir('data\*.mat');
save_path_1 = 'pick_data\';  %after run this code, please  copy this file into anomaly_score file



for i = 1:length(filelist1)
    disp(i);
    a = ['data\' filelist1(i).name];
    file_name = filelist1(i).name;

    load(a);  % load data
    Bottom = interp1(Time, 1:length(Time), Bottom);
    if all(isnan(Bottom))
      continue; % 
    else
        disp('The missing values in the current file were introduced during the pick process.');
        disp(file_name);
        
        bottom = fillmissing(Bottom, 'nearest');  
    end

    Surface = interp1(Time, 1:length(Time), Surface);
    if all(isnan(Surface))
        surface = 300 * ones(size(Surface));
    else
        surface = fillmissing(Surface, 'nearest'); 
    end
    Surface = interp1(GPS_time, surface, GPS_time);  

    IPRData = 10 * log10(Data);
    [m, n] = size(IPRData);
    w = 100; 
    bottom2 = zeros(2*w+1, n);
    fs = 1 / (Time(2) - Time(1));  
    
    S_Bed = [];
    bottommax = [];
    pick_signal = [];


    bottom = round(bottom);

    for j = 1:n
        disp(j);
        bottom2(:, j) = IPRData(ceil(bottom(j))-100:ceil(bottom(j)+100), j);
    end

    current_name = file_name(1:end-4);
   
    pick_signal = bottom2;

    pick_value = pick_signal';
    save_result = [save_path_1, file_name];
    save(save_result, 'pick_value')
    disp(['save already: ', num2str(i)])



end
