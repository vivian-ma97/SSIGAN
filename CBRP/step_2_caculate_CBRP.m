
clc;
clear;

path_1 = 'C:\Users\ma_97\Desktop\干湿代码按照kenny建议讨论后\R3_3\data';  %% cresis data path

save_path_1 = 'C:\Users\ma_97\Desktop\干湿代码按照kenny建议讨论后\R3_3\final_CBRP_chu';  % save CBRP result path

files_1 = dir(fullfile(path_1, '*.mat'));  

if ~exist(save_path_1, 'dir')
    mkdir(save_path_1);
end
 
all_S_Bed = [];

AT = 12.760; % Chu c0=0.7;

for i = 1:length(files_1)
    disp(i);

    file_1 = fullfile(path_1, files_1(i).name);

    disp(['current process data: ', files_1(i).name]);
    
    load(file_1);  

 
    
    Radar_data = lp(Data);    
    Bottom =interp1(Time,1:length(Time),Bottom);
    if all(isnan(Bottom))
    continue; 
    else
        bottom = fillmissing(Bottom,'nearest');
    end
    Bottom = interp1(GPS_time,bottom,GPS_time); 
    bottom = Bottom;
    Surface =interp1(Time,1:length(Time),Surface);
    Surface = interp1(GPS_time,Surface,GPS_time); 
    if all(isnan(Surface))
        surface = 300*ones(size(Surface));
    else
        surface = fillmissing(Surface,'nearest'); 
    end

    dt=Time(2)-Time(1);                         
    [m,n] = size(Radar_data);                      
    w = 150;                                    
    bottom2 = zeros(2*w+1,n);                   
    fs = 1 / dt;                                
    window = 20;                                
    movwindow = 15;                            
    fontsize = 12;                              


    clear bottommax;
    clear S_Bed;


    for j = 1:n
        bottom2(:,j) = Radar_data(ceil(bottom(j))-150:ceil(bottom(j)+150),j);
        bottomtemp(:,j) = Radar_data(ceil(bottom(j))-70:ceil(bottom(j)+70),j);
        [S_Bed(j), bottommax(j)] = max(bottomtemp(:,j));
    end

    bottom = bottommax + bottom -51;
    bottom_aver = movmean(bottom,window);
    s = movmean(Radar_data,window,2);
    c = 299792458;                                     
    er = 1.78^2;                                          
    H = (bottom-surface) * dt * c / sqrt(er) / 2 / 1000;
    h = surface * dt * c / 2 / 1000;                    
    G = 10 * log10(2 * (h*1000 + H*1000 / sqrt(er)));   
    P_Bed = S_Bed + 2 * G + 2 * AT * H;               % final CBRP
    all_S_Bed = [all_S_Bed, P_Bed];

    [~, base_name, ~] = fileparts(files_1(i).name);
    save(fullfile(save_path_1, [base_name, '.mat']), 'P_Bed');
  

end



%  max CBRP is  90.1552

