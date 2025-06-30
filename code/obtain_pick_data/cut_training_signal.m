clc;
clear;

filelist1 = dir('data\*.mat');
save_path_1 = 'pick_data\';  % 保存的pick_文件

LAKE_excel = readtable('training.xlsx');
[m_1, n_1] = size(LAKE_excel); % 正确的语法

lake_mat = [];  % 初始化用于保存湖区信号的矩阵
non_lake_mat = [];  % 初始化用于保存非湖区信号的矩阵

for i = 1:length(filelist1)
    disp(i);
    a = ['data\' filelist1(i).name];
    file_name = filelist1(i).name;

    load(a);  % 加载数据
    Bottom = interp1(Time, 1:length(Time), Bottom);
    if all(isnan(Bottom))
      continue; % 如果全是 NaN，跳过当前循环
    else
        disp('现在的文件中存在空值是通过pick实现的');
        disp(file_name);
        bottom = fillmissing(Bottom, 'nearest');  % 补全bottom
    end

    Surface = interp1(Time, 1:length(Time), Surface);
    if all(isnan(Surface))
        surface = 300 * ones(size(Surface));
    else
        surface = fillmissing(Surface, 'nearest'); % 补全surface
    end
    Surface = interp1(GPS_time, surface, GPS_time);  % 冰层表面

    IPRData = 10 * log10(Data);
    [m, n] = size(IPRData);
    w = 100;  % 定义宽度
    bottom2 = zeros(2*w+1, n);
    fs = 1 / (Time(2) - Time(1));  % 采样频率
    
    S_Bed = [];
    bottommax = [];
    pick_signal = [];

    for j = 1:n
        bottom2(:, j) = IPRData(ceil(bottom(j))-100:ceil(bottom(j)+100), j);
    end

    current_name = file_name(1:end-4);
    label_begin_0 = zeros(n, 1);

    for j = 1:m_1
        if strcmp(current_name, LAKE_excel.Cresis_Frame{j})
            label_lake_idx_begin = LAKE_excel.Creis_Begin(j);
            label_lake_idx_end = LAKE_excel.Cresis_End(j);
            label_begin_0(label_lake_idx_begin:label_lake_idx_end) = 1;
        end
    end

    final_label = label_begin_0;
    pick_signal = bottom2;

    % 根据标签挑选信号
    lake_signal = pick_signal(:, final_label == 1);
    non_lake_signal = pick_signal(:, final_label == 0);

    % 合并信号到对应矩阵
    lake_mat = [lake_mat, lake_signal];
    non_lake_mat = [non_lake_mat, non_lake_signal];

    pick_value = pick_signal';
    save_result = [save_path_1, file_name];
    save(save_result, 'pick_value', 'lake_signal', 'non_lake_signal')
    disp(['save already: ', num2str(i)])
end

% 可选：保存最终的湖区和非湖区信号矩阵
save('lake_signals.mat', 'lake_mat');
save('non_lake_signals.mat', 'non_lake_mat');
