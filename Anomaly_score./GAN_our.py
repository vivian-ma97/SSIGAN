import numpy as np
from tensorflow import keras
from keras.layers import Input, Dense, Reshape, Dropout, Flatten
from keras.layers import Activation, UpSampling1D
# from keras.layers import BatchNormalization

from keras.layers import Conv1DTranspose, Conv1D, Bidirectional, LSTM
from keras.layers import LeakyReLU, MaxPooling1D, Concatenate
from keras.models import Sequential, Model
from keras.optimizers import adam_v2
import matplotlib.pyplot as plt
import os
import pickle
from Minibatchdiscrimination import MinibatchDiscrimination
import h5py, json
import module.generator as Gen
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Concatenate, Input
class DCGAN:

    def __init__(self, input_shape=(301, 1), latent_size=100, random_sine=True, scale=1, minibatch=False,
                 gen_version=0):

        self.gen_v = gen_version
        self.input_shape = input_shape
        self.latent_size = latent_size
        optimizer = adam_v2.Adam(lr=0.0002, beta_1=0.5)
        self.optimizer = optimizer
        self.random_sine = random_sine
        self.scale = scale
        self.minibatch = minibatch
        # build and compile discriminator
        self.discrimintor = self.bulid_discrimintor()
        self.is_first_input = True
        self.discrimintor.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.cov_info = 0
        self.cosine_loss = 0
        # build generator
        self.generator = self.build_generator()

        # generator takes noise as input and generates signals
        z = Input(shape=(self.latent_size,))
        signal = self.generator(z)


        # for combined model, we only train the generator
        # for combined model, we only train the generator
        self.discrimintor.trainable = False

        # discrimator takes generate signals as input and determines validity
        valid = self.discrimintor(signal)


        # combine model, stack generator and discriminator
        # train the generator to fool discriminator
        self.combine = Model(z, valid)
        self.combine.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        if (self.gen_v == 0 or self.gen_v == None):
            model = Sequential(name='Generator')
            model.add(Reshape((self.latent_size, 1)))
            model.add(Bidirectional(LSTM(16, return_sequences=True)))

            model.add(Conv1D(32, kernel_size=8, padding="same"))
            model.add(LeakyReLU(alpha=0.2))

            model.add(UpSampling1D())
            model.add(Conv1D(16, kernel_size=8, padding="same"))
            model.add(LeakyReLU(alpha=0.2))

            model.add(UpSampling1D())
            model.add(Conv1D(8, kernel_size=8, padding="same"))
            model.add(LeakyReLU(alpha=0.2))

            model.add(Conv1D(1, kernel_size=8, padding="same"))
            model.add(Flatten())

            model.add(Dense(self.input_shape[0]))
            model.add(Activation('tanh'))
            model.add(Reshape(self.input_shape))
            noise = Input(shape=(self.latent_size,))
            signal = model(noise)

            model.summary()

            return Model(inputs=noise, outputs=signal)

        elif self.gen_v in [1, 2, 3, 4, 5]:  # use different generater from in_progress
            model = Gen.Generator(self.latent_size, self.input_shape)

            if self.gen_v == 1:
                return model.G_vl()
            if self.gen_v == 2:
                return model.G_v2()
            if self.gen_v == 3:
                return model.G_v3()
            if self.gen_v == 4:
                return model.G_v4()
            if self.gen_v == 5:
                return model.G_v5()

        else:
            raise ValueError("Invalid generator version.")

    def bulid_discrimintor(self):

        signal = Input(shape=self.input_shape)

        if self.minibatch:

            flat = Flatten()(signal)
            mini_disc = MinibatchDiscrimination(10, 3)(flat)

            md = Conv1D(8, kernel_size=8, strides=1, input_shape=self.input_shape, padding='same')(signal)
            md = LeakyReLU(alpha=0.2)(md)
            md = Dropout(0.25)(md)
            md = MaxPooling1D(3)(md)

            md = Conv1D(16, kernel_size=8, strides=1, input_shape=self.input_shape, padding='same')(md)
            md = LeakyReLU(alpha=0.2)(md)
            md = Dropout(0.25)(md)
            md = MaxPooling1D(3, strides=2)(md)

            md = Conv1D(32, kernel_size=8, strides=2, input_shape=self.input_shape, padding='same')(md)
            md = LeakyReLU(alpha=0.2)(md)
            md = Dropout(0.25)(md)
            md = MaxPooling1D(3, strides=2)(md)

            md = Conv1D(64, kernel_size=8, strides=2, input_shape=self.input_shape, padding='same')(md)
            md = LeakyReLU(alpha=0.2)(md)
            md = Dropout(0.25)(md)
            md = MaxPooling1D(3, strides=2)(md)
            md = Flatten()(md)
            concat = Concatenate()([md, mini_disc])
            validity = Dense(1, activation='sigmoid')(concat)
            md.summary()

            return Model(inputs=signal, outputs=validity, name="Discriminator")
            # return Model(inputs=signal, outputs=validity)



        else:
            model = Sequential(name='Discriminator')
            # model = Sequential()
            model.add(Conv1D(8, kernel_size=8, strides=1, input_shape=self.input_shape, padding='same'))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.25))
            model.add(MaxPooling1D(3))

            model.add(Conv1D(16, kernel_size=8, strides=1, input_shape=self.input_shape, padding='same'))
            model.add(Dropout(0.25))
            model.add(MaxPooling1D(3, strides=2))

            model.add(Conv1D(32, kernel_size=8, strides=2, input_shape=self.input_shape, padding='same'))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.25))
            model.add(MaxPooling1D(3, strides=2))

            model.add(Conv1D(64, kernel_size=8, strides=2, input_shape=self.input_shape, padding='same'))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.25))
            model.add(MaxPooling1D(3, strides=2))

            model.add(Flatten())
            model.add(Dense(1, activation='sigmoid'))
            validity = model(signal)

            model.summary()



            return Model(inputs=signal, outputs=validity)

    def train(self, epochs, X_train, batch_size=2, save_interval=50, save=False, save_model_interval=100,
              save_report=True):
        vaild = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        progress = {'d_loss': [],
                    'g_loss': [],
                    'acc': []}
        train_loss = []
        train_accuracy = []
        for epoch in range(epochs):

            # -------------------
            # Train discriminator
            # -------------------

            # select a random batch of signals
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            signals = X_train[idx]

            if self.is_first_input:
                noise = self.generate_noise(batch_size, self.random_sine)
                self.is_first_input = False
            else:

                noise = self.cov_info * self.generate_noise(batch_size, self.random_sine)

            # sample noise and generatir a batch of new signals

            gen_signals = self.generator.predict(noise)


            gen_signals_flatten = gen_signals.flatten()
            signals_flatten = signals.flatten()


            self.cov_info = np.cov(gen_signals_flatten, signals_flatten)[0, 1]

            # caculate cosin loss
            self.cosine_loss = tf.losses.cosine_distance(signals_flatten, gen_signals_flatten, axis=0)

            # train the discriminator (real signals labeled as 1 and fake labeled as 0)
            d_loss_real = self.discrimintor.train_on_batch(signals, vaild)
            d_loss_fake = self.discrimintor.train_on_batch(gen_signals, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # -------------------
            # Train Generator
            # -------------------

            # Train the generator (Goal: fool discriminator)
            g_loss = self.combine.train_on_batch(noise, vaild)+ self.cov_info* self.cosine_loss

            # print the progresss
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # save progress
            progress['d_loss'].append(d_loss[0])
            progress['acc'].append(d_loss[1])
            progress['g_loss'].append(g_loss)

            train_loss.append(d_loss[0])
            train_accuracy.append(100 * d_loss[1])

            # if reach save interval, plot the signals and save as image
            if epoch % save_interval == 0:
                self.save_image(epoch)
            if save:
                if os.path.isdir('save_cov_model/') != True:
                    os.mkdir('save_cov_model/')
                if (epoch % save_model_interval == 0 and epoch > 0):
                    self.generator.save('save_cov_model/gen_%d.h5' % epoch)
                    self.discrimintor.save('save_cov_model/dis_%d.h5' % epoch)
                    self.save_sample(self.generator, self.discrimintor, batch_size, epoch)


        # save last round result
        self.save_image(epoch)
        self.generator.save('save_cov_model/gen_%d.h5' % epoch)
        self.save_sample(self.generator, self.discrimintor, batch_size, epoch)



        # save progress report
        # progress['variable'] = {"optimizer":self.optimizer.name}
        if save_report:
            with open('output_cov/progress_report.json', 'w') as f:
                json.dump(progress, f)



        plt.close()
        plt.plot(np.arange(len(train_loss)), train_loss, label='train loss')
        # plt.plot(np.arange(len(train_accuracy)), train_accuracy, label='train accuracy')
        plt.title('Model loss')
        plt.xlabel('Epoch')
        plt.legend(['Train_loss'])
        plt.show()

    def save_image(self, epoch):

        if os.path.isdir('image_cov/') != True:
            os.mkdir('image_cov/')

        r, c = 3, 3
        noise = self.generate_noise(r * c, self.random_sine)

        signals = self.generator.predict(noise) * self.scale

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].plot(signals[cnt])
                cnt += 1

        fig.savefig('image_cov/ecg_sig_%d.png' % epoch)
        plt.close()

    def prepare_input(self, dataset):

        if type(dataset) != dict:
            raise TypeError('Dateset type must be dictionary.')
        X_train = []
        y = []
        for sg_id in dataset.keys():
            lb = str(dataset[sg_id][0])
            if lb != '~':
                signal = dataset[sg_id][1]
                for hb in signal:
                    X_train.append(hb)
                    y.append(lb)

        X_train = np.array(X_train).reshape(-1, X_train.shape[1], 1)
        y = np.array(y)
        return X_train, y

    def generate_noise(self, batch_size, sinwave=False):
        '''
        generate noise
        if sinwave is True, generate sin wave noise, otherwise, return standard normal distribution.
        '''
        if sinwave:
            x = np.linspace(-np.pi, np.pi, self.latent_size)
            noise = 0.1 * np.random.random_sample((batch_size, self.latent_size)) + 0.9 * np.sin(x)
        else:
            noise = np.random.normal(0, 1, size=(batch_size, self.latent_size))
        return noise

    def specify_range(self, signals, min_val=-2, max_val=2):
        """
        Specify acceptable range, drop signal if signal value is out of range.
        """

        if signals is None:
            raise ValueError("No signals data.")
        if type(signals) != np.ndarray:
            signals = np.array(signals)
        select_signals = []
        for sg in signals:
            min_sg = np.min(sg)
            max_sg = np.max(sg)

            if (min_sg >= min_val and max_sg <= max_val):
                select_signals.append(sg)

        return np.array(select_signals)

    def save_sample(self, generator, discriminator, sample_size, epoch):
        '''
        generate signal samples and save as .csv
        '''
        if os.path.isdir('output_cov/') != True:
            os.mkdir('output_cov/')

        # generate signals
        noise = self.generate_noise(sample_size)
        signals = generator.predict(noise)

        # classify the signals is real or fake
        critic = discriminator.predict(signals)

        # recover signal by mutiplying scale
        signals = signals.reshape(-1, self.input_shape[0]) * self.scale

        valid_sample = []
        for i, decision in enumerate(critic):
            if decision > 0.5:
                valid_sample.append(signals[i])

        np.savetxt('output_cov/sample_%d.csv' % epoch, valid_sample, delimiter=',')
        np.savetxt('output_cov/prob_%d.csv' % epoch, critic, delimiter=',')

    def rescale_signal(self, signals, min_val=-1, max_val=1):
        """
        :param signals:
        :param min_val:
        :param max_val:
        :return:
        """

        max_val = np.max(signals)
        min_val = np.min(signals)
        scale = max_val - min_val
        scale_signal = (signals - min_val) / scale
        return scale_signal, scale










