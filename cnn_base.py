# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 20:20:02 2020

@author: rasool
"""

from __future__ import division
from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, AvgPool2D
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class CNN (object):
    def __init__(self, num_classes, input_shape, node_type='root', ds_name='cifar10'):
        # input_layer, l = self.create_network_base(num_classes, input_shape)
        self.num_classes = num_classes
        self.input_shape = input_shape       
        self.dataset_name = ds_name
       
        # if node_type == 'root':
        #     self.model = self.create_root_model_paper(self.num_classes)
        # else:
        #     self.model = self.create_branch_model_paper(self.num_classes)
        if ds_name == 'cifar10':
            self.model = self.create_model_cf10(self.num_classes)
        else:
            self.model = self.create_model_cf100(self.num_classes)

        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    def create_model_cf10(self, num_clss):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=self.input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        # model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        # model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        # model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        # model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_clss, activation='relu'))
        # model.add(Activation('softmax'))
        model.summary()
        return model

    def create_model_cf100(self, num_clss):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), padding='same', input_shape=self.input_shape))
        model.add(Activation('elu'))
        model.add(Conv2D(64, (3, 3), activation='elu'))
        # model.add(Activation('elu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.1))

        model.add(Conv2D(128, (3, 3), padding='same', activation='elu'))
        # model.add(Activation('elu'))
        model.add(Conv2D(128, (3, 3), activation='elu'))
        # model.add(Activation('elu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (3, 3), padding='same', activation='elu'))
        # model.add(Activation('elu'))
        model.add(Conv2D(256, (3, 3), activation='elu'))
        # model.add(Activation('elu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(1024, activation='elu'))
        # model.add(Activation('elu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_clss, activation='relu'))
        # model.add(Activation('softmax'))
        model.summary()
        return model

    def create_root_model_paper(self, num_clss):
        model = Sequential()
        # Define root node of the tree
        #model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
        model.add(Conv2D(64, (5, 5), input_shape=self.input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        if self.dataset_name == 'cifar10':
            model.add(Flatten())
            model.add(Dense(512, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(num_clss, activation='relu'))
            # softmax layer ????
        else:
            model.add(Conv2D(256, (3,3), activation='relu'))
            model.add(Dropout(0.5))
            #model.add(Conv2D(256, (3,3), activation='relu'))
            model.add(AvgPool2D(pool_size=(2, 2)))

            model.add(Flatten())
            model.add(Dense(1024, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1024, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(num_clss, activation='relu'))
            # softmax layer ????
        model.summary()
        return model

    def create_branch_model_paper(self, num_clss):
        model = Sequential()
        model.add(Conv2D(32, (5, 5), input_shape=self.input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
            
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
           
        if self.dataset_name == 'cifar10':
            model.add(Dropout(0.25))
            model.add(Conv2D(64, (3,3), activation='relu'))
            model.add(AvgPool2D(pool_size=(2, 2)))
                
            model.add(Dropout(0.25))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(num_clss, activation='relu'))   
        else:   
            model.add(Dropout(0.25))
            model.add(Conv2D(64, (3,3), activation='relu'))
            # model.add(Dropout(0.5))
            # model.add(Conv2D(64, (3,3), activation='relu'))
            model.add(AvgPool2D(pool_size=(2, 2)))
                
            model.add(Flatten())
            model.add(Dense(512, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(num_clss, activation='relu'))

        model.summary()
        return model

    def train(self, X, Y, x_test, y_test):
        # cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)
        # print(X.shape, Y.shape)
        # self.model.fit(X, Y, batch_size=16, epochs=50, validation_split=0.1,shuffle=True)
        # #self.model.fit(X, Y, batch_size=16, epochs=300, callbacks=[cb], validation_split=0.1)
        
        batch_size = 10
        epochs = 50
        self.model.fit(X, Y,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)
        
        # print('Using real-time data augmentation.')
        # # This will do preprocessing and realtime data augmentation:
        # datagen = ImageDataGenerator(
        # featurewise_center=False,  # set input mean to 0 over the dataset
        # samplewise_center=False,  # set each sample mean to 0
        # featurewise_std_normalization=False,  # divide inputs by std of the dataset
        # samplewise_std_normalization=False,  # divide each input by its std
        # zca_whitening=False,  # apply ZCA whitening
        # rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        # height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        # horizontal_flip=True,  # randomly flip images
        # vertical_flip=False)  # randomly flip images
        #
        # # Compute quantities required for feature-wise normalization
        # # (std, mean, and principal components if ZCA whitening is applied).
        # datagen.fit(X)
        #
        # # Fit the model on the batches generated by datagen.flow().
        # self.model.fit_generator(datagen.flow(X, Y,
        #                                       batch_size=batch_size),
        #                          steps_per_epoch=X.shape[0] // 10,
        #                          epochs=epochs,
        #                          validation_data=(x_test, y_test))

    def remove_class(self, idx_to_remove):
#         input_layer, l = self.create_network_base((self.num_classes-1), self.input_shape)
       # new_model = tf.keras.applications.resnet.ResNet50(include_top=True, weights=None, classes=(self.num_classes-1), input_shape=self.input_shape)
#     Model(input_layer, l)
        # root 13 cif10 , 16 cif100      branch: cif10 14 , cif100 15
        # if len(self.model.layers) == 13 or len(self.model.layers) == 16:
        #     new_model = self.create_root_model(self.num_classes - 1)
        # else:
        #     new_model = self.create_branch_model(self.num_classes - 1)
        if self.dataset_name == 'cifar10':
            new_model = self.create_model_cf10(self.num_classes - 1)
        else:
            new_model = self.create_model_cf100(self.num_classes - 1)

        for idx in range(len(self.model.layers)-1):
            if len(self.model.layers[idx].get_weights()) == 0 :
                continue
            if len(self.model.layers[idx].get_weights()) == 2 :
                wi = self.model.layers[idx].get_weights()[0]
                bi = self.model.layers[idx].get_weights()[1]
                new_model.layers[idx].set_weights((wi, bi))
            if len(self.model.layers[idx].get_weights()) == 4 :
                wi = self.model.layers[idx].get_weights()[0]
                bi = self.model.layers[idx].get_weights()[1]
                wi2 = self.model.layers[idx].get_weights()[2]
                bi2 = self.model.layers[idx].get_weights()[3]
                new_model.layers[idx].set_weights((wi, bi, wi2, bi2))

        # Copy an already trained part of last layer
        old_w = self.model.layers[-1].get_weights()[0]
        new_w = new_model.layers[-1].get_weights()[0]
        old_bias = self.model.layers[-1].get_weights()[1]
        new_bias = new_model.layers[-1].get_weights()[1]

        for i in range(old_w.shape[0]):
            aux = 0
            for j in range(old_w.shape[1]):
                if j != idx_to_remove:
                    new_w[i][aux] = old_w[i][j]
                    aux = aux + 1
        aux = 0
        for i in range(old_bias.shape[0]):
            if i != idx_to_remove:
                new_bias[aux] = old_bias[i]
                aux = aux + 1

        new_model.layers[-1].set_weights((new_w, new_bias))

        self.model = new_model
        #self.model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])  
        self.num_classes = self.num_classes - 1
        
    def add_class(self):
#         input_layer, l = self.create_network_base((self.num_classes+1), self.input_shape)
        #new_model = tf.keras.applications.resnet.ResNet50(include_top=True, weights=None, classes=(self.num_classes+1), input_shape=self.input_shape)
#     Model(input_layer, l)
        # root 13 cif10 , 16 cif100      branch: cif10 14 , cif100 15
        # if len(self.model.layers) == 13 or len(self.model.layers) == 16:
        #     new_model = self.create_root_model(self.num_classes + 1)
        # else:
        #     new_model = self.create_branch_model(self.num_classes + 1)
        if self.dataset_name == 'cifar10':
            new_model = self.create_model_cf10(self.num_classes + 1)
        else:
            new_model = self.create_model_cf100(self.num_classes + 1)
        
        for idx in range(len(self.model.layers)-1):
            if len(self.model.layers[idx].get_weights()) == 0 :
                continue
            if len(self.model.layers[idx].get_weights()) == 2 :
                wi = self.model.layers[idx].get_weights()[0]
                bi = self.model.layers[idx].get_weights()[1]
                new_model.layers[idx].set_weights((wi, bi))
            if len(self.model.layers[idx].get_weights()) == 4 :
                wi = self.model.layers[idx].get_weights()[0]
                bi = self.model.layers[idx].get_weights()[1]
                wi2 = self.model.layers[idx].get_weights()[2]
                bi2 = self.model.layers[idx].get_weights()[3]
                new_model.layers[idx].set_weights((wi, bi, wi2, bi2))

        # Copy an already trained part of last layer
        old_w = self.model.layers[-1].get_weights()[0]
        new_w = new_model.layers[-1].get_weights()[0]
        old_bias = self.model.layers[-1].get_weights()[1]
        new_bias = new_model.layers[-1].get_weights()[1]

        for i in range(old_w.shape[0]):
            for j in range(old_w.shape[1]):
                new_w[i][j] = old_w[i][j]
        for i in range(old_bias.shape[0]):
            new_bias[i] = old_bias[i]

        new_model.layers[-1].set_weights((new_w, new_bias))

        self.model = new_model
        #self.model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])  
        self.num_classes = self.num_classes + 1
        
    def pred(self, img):
        return self.model.predict(img)

