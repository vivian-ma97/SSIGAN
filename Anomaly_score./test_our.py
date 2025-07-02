"""
加载模型进行异常检测

cov_loss


"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import os
from scipy.io import loadmat, savemat

from GAN_our import *

def generate_noise(latent_size, batch_size, sinwave=False):
    '''
    generate noise
    if sinwave is True, generate sin wave noise, otherwise, return standard normal distribution.
    '''
    if sinwave:
        x = np.linspace(-np.pi, np.pi, latent_size)
        noise = 0.1 * np.random.random_sample((batch_size, latent_size)) + 0.9 * np.sin(x)
    else:
        noise = np.random.normal(0, 1, size=(batch_size, latent_size))
    return noise


import numpy as np

def anomaly_score(input_signal, fake_signal, discriminator):
    # 将 input_signal 转换为 NumPy 数组
    input_signal = np.array(input_signal[0], dtype=np.float64)

    # 调整 fake_signal 的形状
    fake_signal = np.squeeze(fake_signal, axis=-1)

    # 计算总体损失
    total_loss = np.mean(np.abs(input_signal - fake_signal), axis=1)

    # 计算判别器损失
    discriminator_loss = -np.mean(discriminator(fake_signal))

    # 计算残差损失
    residual_loss = np.sum(np.abs(np.squeeze(input_signal) - fake_signal), axis=1)

    return total_loss, discriminator_loss, residual_loss

def rescale_signal(signals, min_val=-1, max_val=1):
    """
    对信号进行归一化
    :param signals:
    :param min_val:
    :param max_val:
    :return:
    """

    signals = np.array(signals)
    max_val = np.max(signals)
    min_val = np.min(signals)
    scale = max_val - min_val
    scale_signal = (signals - min_val) / scale
    return scale_signal, scale



latent_size = 100
batch_size = 64
random_sine = True


# 加载生成器模型和判别器模型
generator = load_model('save_model/gen_7400.h5')  # 根据实际情况替换为您的生成器模型文件路径

noise = generate_noise(latent_size, batch_size, random_sine)
gen_signals = generator.predict(noise)

# 加载生成器模型和判别器模型
d = load_model('save_model/dis_7400.h5')  # 根据实际情况替换为您的生成器模型文件路径


# 创建结果保存目录
result_dir = 'Anomaly_score'
os.makedirs(result_dir, exist_ok=True)


file_path = os.listdir('pick_data3')


     
# 逐个处理.mat文件并转换为.pkl文件
for idx, mat_file in enumerate(file_path):
    # 首先将mat文件换成pkl

    mat_data = loadmat('pick_data3/' + mat_file)
    pick_data = mat_data['pick_value']

    new_file_name = mat_file[:-4] + '.pkl'
    # 保存为.pkl文件
    with open(new_file_name, 'wb') as f:
        pickle.dump(pick_data, f)

    signal = pickle.load(open(new_file_name, 'rb'))

    signal = np.array(signal)

    # 异常检测的测试逻辑
    anomaly_scores = []  # 存储异常评分
    batch_size = 1 # 设置批处理大小

    num_samples = len(signal)  # 信号样本数量
    num_batches = num_samples // batch_size  # 计算批次数量

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size

        # 提取批次信号数据
        batch_signal = signal[start_idx:end_idx]


        noise = generate_noise(latent_size, batch_size, sinwave=True)

        batch_signal = rescale_signal(signals=batch_signal)

        # 生成fake signal
        gen_signals = generator.predict(noise)  # 根据您的具体生成方法进行修改

        # 计算异常评分
        batch_total_loss, _, batch_residual_loss = anomaly_score(batch_signal, gen_signals, d)

        anomaly_scores.extend(batch_total_loss)


    # 处理剩余的不满批次的信号数据
    remaining_samples = num_samples % batch_size
    if remaining_samples > 0:
        start_idx = num_batches * batch_size
        batch_signal = signal[start_idx:]
        noise = generate_noise(latent_size, batch_size, sinwave=True)

        batch_signal = rescale_signal(signals=batch_signal)

        # 生成fake signal
        gen_signals = generator.predict(noise)  # 根据您的具体生成方法进行修改

        # 计算异常评分
        batch_total_loss, _, batch_residual_loss = anomaly_score(batch_signal, gen_signals, d)

        anomaly_scores.extend(batch_total_loss)

    # 构建结果文件路径

    # 将异常得分列表转换为NumPy数组
    anomaly_scores_array = np.array(anomaly_scores)
    print('current_mat_file_name:', mat_file)
    # 保存异常得分数组为MAT文件
    result_file_path = os.path.join(result_dir, mat_file)
    savemat(result_file_path, {'anomaly_scores': anomaly_scores_array})
