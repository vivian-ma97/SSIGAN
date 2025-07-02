function [global_min, global_max] = find_mat_min_max(folder_path)
% FIND_MAT_MIN_MAX 遍历文件夹中的所有.mat文件，找出所有数据中的最小值和最大值
%
% 输入参数:
%   folder_path - 包含.mat文件的文件夹路径(字符串)
%
% 输出参数:
%   global_min - 所有.mat文件中的最小值
%   global_max - 所有.mat文件中的最大值

% 初始化全局最小值和最大值
global_min = Inf;
global_max = -Inf;



folder_path = 'C:\Users\ma_97\Desktop\干湿代码按照kenny建议讨论后\all_agap_anomaly_score\Anomaly_score';

% 获取文件夹中所有.mat文件
file_list = dir(fullfile(folder_path, '*.mat'));

% 检查是否找到.mat文件
if isempty(file_list)
    error('在指定文件夹中未找到任何.mat文件: %s', folder_path);
end

% 遍历每个.mat文件
for i = 1:length(file_list)
    file_name = fullfile(folder_path, file_list(i).name);
    
    try
        % 加载.mat文件中的所有变量
        file_data = load(file_name);
        
        % 获取文件中的所有变量名
        vars = fieldnames(file_data);
        
        % 遍历每个变量
        for j = 1:length(vars)
            current_var = file_data.(vars{j});
            
            % 只处理数值型数据(忽略其他类型如字符串、结构体等)
            if isnumeric(current_var)
                % 更新全局最小值和最大值
                current_min = min(current_var(:));
                current_max = max(current_var(:));
                
                if current_min < global_min
                    global_min = current_min;
                end
                
                if current_max > global_max
                    global_max = current_max;
                end
            end
        end
        
    catch ME
        warning('无法处理文件 %s: %s', file_name, ME.message);
        continue;
    end
end

% 如果没有找到有效的数值数据
if isinf(global_min) || isinf(global_max)
    error('在.mat文件中未找到任何数值数据');
end

fprintf('处理完成。共处理了 %d 个.mat文件。\n', length(file_list));
fprintf('全局最小值: %g\n', global_min);
fprintf('全局最大值: %g\n', global_max);
end


