import os

import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, DepthwiseConv2D, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import sigmoid

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec

from keras import backend as K 
from tensorflow.keras.saving import get_custom_objects
# from bcnn_utils import img_gen
# from bayesian_NN_experiments.bcnn_classifier import bcnn_utils as bcu


tfd = tfp.distributions
tfpl = tfp.layers


trimg_dir = "/Users/karljaehnig/archive/seg_train/seg_train/"
tsimg_dir = "/Users/karljaehnig/archive/seg_test/seg_test/"


def swish(x):
    return (K.sigmoid(x)*x)

get_custom_objects().update({'swish':Activation(swish)})

def neg_loglike(ytrue, ypred):
    return -ypred.log_prob(ytrue)

def divergence(q,p,_):
    return tfd.kl_divergence(q,p)/14034


def bmodel(): 

    mdl = Sequential(name='sequential_14')

    get_custom_objects().update({'swish':Activation(swish)})

    mdl.add(get_convolutional_reparameterization_layer(
        input_shape=(150,150,3), 
        divergence_fn=divergence,
        filters=32,
        name='conv2d_reparameterization_2'
        )
    )


    mdl.add(MaxPooling2D(2,2,name='max_pooling2d_47'))

    mdl.add(DepthwiseConv2D(
        # filters=16,
        depth_multiplier=4,
        kernel_size=(3,3),
        activation='swish',
        kernel_initializer='normal',
        padding='Same',
        name='depthwise_conv2d_13'
        )
    ) 

    mdl.add(MaxPooling2D(2,2,name='max_pooling2d_48'))
    mdl.add(Dropout(0.25,name='dropout_36'))

    mdl.add(Conv2D(
        filters=64,
        kernel_size=(3,3),
        activation='swish',
        kernel_initializer='normal',
        padding='Same',
        name='conv2d_33'
        ))

    mdl.add(MaxPooling2D(2,2,name='max_pooling2d_49'))
    mdl.add(Dropout(0.25,name='dropout_37'))


    mdl.add(Conv2D(
        filters=64,
        kernel_size=(3,3),
        activation='swish',
        kernel_initializer='normal',
        padding='Same',
        name='conv2d_34'
        ))

    mdl.add(MaxPooling2D(2,2,name='max_pooling2d_50'))
    mdl.add(Dropout(0.25,name='dropout_38'))


    mdl.add(Conv2D(
        filters=128,
        kernel_size=(3,3),
        activation='swish',
        kernel_initializer='normal',
        padding='Same',
        name='conv2d_35'
        )
    ) 

    mdl.add(MaxPooling2D(2,2,name='max_pooling2d_51'))
    mdl.add(Dropout(0.25,name='dropout_39'))

    mdl.add(Conv2D(
        filters=128,
        kernel_size=(3,3),
        activation='swish',
        kernel_initializer='normal',
        padding='Same',
        name='conv2d_36'
        ))


    mdl.add(Conv2D(
        filters=128,
        kernel_size=(3,3),
        activation='swish',
        kernel_initializer='normal',
        padding='Same',
        name='conv2d_37'
        ))


    mdl.add(MaxPooling2D(2,2,name='max_pooling2d_52'))
    mdl.add(Dropout(0.75,name='dropout_40'))
    mdl.add(Flatten(name='flatten_7'))

    mdl.add(get_dense_reparameterization_layer(divergence,name='dense_reparameterization_2'))
    mdl.add(tfpl.OneHotCategorical(6,
            convert_to_tensor_fn=tfd.Distribution.mode,
            name='one_hot_categorical_7'))

    return mdl




def main():
    trgen,tsgen = bcu.img_gen(trimg_dir,tsimg_dir)

    cm_callback = keras.callbacks.LambdaCallback(
        on_epoch_end=bcu.log_confusion_matrix)

    mdl = model()

    mdl.summary()

    mdl.compile(
        loss=neg_loglike,
        optimizer=Adam(learning_rate=0.005),
        metrics=['accuracy'],
        experimental_run_tf_function=False
        )

    mdl_hist = mdl.fit(
        trgen,
        epochs=200,
        verbose=1,
        validation_data=tsgen,
        callbacks=['cm_callback']
        )


# mdl.predict(tsgen)

