# -*- coding: utf-8 -*-
"""
Created by etayupanta at 6/30/2020 - 21:10
PyCharm - DeepVO
__author__ = 'Eduardo Tayupanta'
__email__ = 'etayupanta@yotec.tech'
"""

# Import Libraries:
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DeepVONet:
    def __init__(self, args, height, width):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

        self.mode = args.mode
        self.datapath = args.datapath
        self.bsize = args.bsize
        self.lr = args.lr
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.tau = args.tau
        self.debug = args.debug
        self.train_iter = args.train_iter
        self.validation_steps = args.validation_steps
        self.epsilon = args.epsilon
        self.checkpoint_path = args.checkpoint_path
        self.checkpoint = args.checkpoint

        self.k = 100

        conv1 = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation=tf.nn.relu,
                              input_shape=(height * 2, width, 3))
        conv2 = layers.Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same', activation=tf.nn.relu)
        conv3 = layers.Conv2D(256, kernel_size=(5, 5), strides=(2, 2), padding='same', activation=tf.nn.relu)
        conv3_1 = layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu)
        conv4 = layers.Conv2D(512, kernel_size=(3, 3), strides=(2, 2), padding='same', activation=tf.nn.relu)
        conv4_1 = layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu)
        conv5 = layers.Conv2D(512, kernel_size=(3, 3), strides=(2, 2), padding='same', activation=tf.nn.relu)
        conv5_1 = layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu)
        conv6 = layers.Conv2D(1024, kernel_size=(3, 3), strides=(2, 2), padding='same', activation=tf.nn.relu)
        reshape = keras.layers.Reshape((-1, 1024))
        lstm1 = layers.LSTM(1000, return_sequences=True)
        lstm2 = layers.LSTM(1000)
        fc = layers.Dense(6)

        self.model = keras.Sequential([
            conv1,
            conv2,
            conv3,
            conv3_1,
            conv4,
            conv4_1,
            conv5,
            conv5_1,
            conv6,
            reshape,
            lstm1,
            lstm2,
            fc
        ])

        self.mse = tf.keras.losses.MeanSquaredError()

    def summary(self):
        return self.model.summary()

    def custom_loss(self, y_true, y_pred):
        mse_position = self.mse(y_true[:3], y_pred[:3])
        mse_orientation = self.mse(y_true[3:], y_pred[3:])
        return mse_position + self.k * mse_orientation

    def compile(self):
        return self.model.compile(
            loss=self.custom_loss,
            optimizer=tf.keras.optimizers.SGD(learning_rate=self.lr),
            metrics=['accuracy']
        )

    def train(self, dataset, train_images_list):
        steps_per_epc = train_images_list / self.bsize

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path + '/cp.ckpt',
                                                         save_weights_only=True,
                                                         verbose=1)

        return self.model.fit(dataset,
                              epochs=self.train_iter,
                              steps_per_epoch=steps_per_epc,
                              callbacks=[cp_callback])
