from utils.data_util import *
#from GAN import *

from GAN_our import *
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pickle
import tensorflow as tf

if __name__ == '__main__':
    data = loadmat('lake_signals.mat')
    data = data['lake_mat']


    # 保存为.pkl文件
    with open('X_train_radar.pkl', 'wb') as f:
        pickle.dump(data, f)

    x_train = pickle.load(open('X_train_radar.pkl', 'rb'))

    x_train = np.transpose(x_train)


        # Setting training phase
    Epoches = 10000
    Save_interval = 100  # 每隔多少次保存一张图片
    Save_model_interval = 100
    Batch_size = 1
    # Input shape
    Input_shape = (201, 1)
    Random_size = False
    Scale = 2
    Mini_batch = False  # use minibatch discriminator to avoid mode collapse
    Save_model = True
    Save_report = True
    dcgan = DCGAN(Input_shape, latent_size=100, random_sine=Random_size, scale=Scale, minibatch=Mini_batch)
    x_train, _ = dcgan.rescale_signal(x_train)
    # print('x_train:',x_train)
    # print(x_train.shape)
    x_train = x_train.reshape(-1, Input_shape[0], Input_shape[1])
    print('Begin to train')
    dcgan.train(Epoches, x_train, Batch_size, Save_interval, save=Save_model, save_model_interval=Save_model_interval,
                save_report=Save_report)
    print('complete!!!!')
