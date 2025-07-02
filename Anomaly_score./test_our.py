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

    input_signal = np.array(input_signal[0], dtype=np.float64)


    fake_signal = np.squeeze(fake_signal, axis=-1)


    total_loss = np.mean(np.abs(input_signal - fake_signal), axis=1)


    discriminator_loss = -np.mean(discriminator(fake_signal))


    residual_loss = np.sum(np.abs(np.squeeze(input_signal) - fake_signal), axis=1)

    return total_loss, discriminator_loss, residual_loss

def rescale_signal(signals, min_val=-1, max_val=1):
    """

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



generator = load_model('save_model/gen_7400.h5')

noise = generate_noise(latent_size, batch_size, random_sine)
gen_signals = generator.predict(noise)


d = load_model('save_model/dis_7400.h5')



result_dir = 'Anomaly_score'
os.makedirs(result_dir, exist_ok=True)


file_path = os.listdir('test_data')


     

for idx, mat_file in enumerate(file_path):

    mat_data = loadmat('test_data/' + mat_file)
    pick_data = mat_data['pick_value']

    new_file_name = mat_file[:-4] + '.pkl'

    with open(new_file_name, 'wb') as f:
        pickle.dump(pick_data, f)

    signal = pickle.load(open(new_file_name, 'rb'))

    signal = np.array(signal)


    anomaly_scores = []
    batch_size = 1

    num_samples = len(signal)
    num_batches = num_samples // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size


        batch_signal = signal[start_idx:end_idx]


        noise = generate_noise(latent_size, batch_size, sinwave=True)

        batch_signal = rescale_signal(signals=batch_signal)


        gen_signals = generator.predict(noise)


        batch_total_loss, _, batch_residual_loss = anomaly_score(batch_signal, gen_signals, d)

        anomaly_scores.extend(batch_total_loss)



    remaining_samples = num_samples % batch_size
    if remaining_samples > 0:
        start_idx = num_batches * batch_size
        batch_signal = signal[start_idx:]
        noise = generate_noise(latent_size, batch_size, sinwave=True)

        batch_signal = rescale_signal(signals=batch_signal)


        gen_signals = generator.predict(noise)
        batch_total_loss, _, batch_residual_loss = anomaly_score(batch_signal, gen_signals, d)

        anomaly_scores.extend(batch_total_loss)


    anomaly_scores_array = np.array(anomaly_scores)
    print('current_mat_file_name:', mat_file)
    result_file_path = os.path.join(result_dir, mat_file)
    savemat(result_file_path, {'anomaly_scores': anomaly_scores_array})
