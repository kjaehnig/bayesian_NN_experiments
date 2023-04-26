import os

import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np 
import matplotlib.pyplot as plt 

from bcnn_utils import img_gen


tfd = tfp.distributions
tfpl = tfp.layers


trimg_dir = "/Users/karljaehnig/archive/seg_train/seg_train/"
tsimg_dir = "/Users/karljaehnig/archive/seg_test/seg_test/"

trgen,tsgen = img_gen(trimg_dir,tsimg_dir)


def neg_loglike(ytrue, ypred):
    return -ypred.log_prob(ytrue)

def divergence(q,p,_):
    return tfd.kl_divergence(q,p)/14034


def model():

    mdl = Sequential()
    mdl.add(
        tfpl.Convolution2DReparameterization(
            input_shape=(150,150,3),
            filters=8,
            kernel_size=16,
            activation='relu',
            kernel_prior_fn=tfpl.default_multivariate_normal_fn,
            kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            kernel_divergence_fn=divergence,
            bias_prior_fn=tfpl.default_multivariate_normal_fn,
            bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            bias_divergence_fn=divergence
            )
        )
    mdl.add(MaxPooling2D(2,2))
    
    mdl.add(Conv2D(16, (5,5), activation='relu'))
    mdl.add(MaxPooling2D(2,2))

    mdl.add(Conv2D(32, (5,5), activation='relu'))
    mdl.add(MaxPooling2D(2,2))

    mdl.add(Conv2D(64, (5,5), activation='relu'))
    mdl.add(MaxPooling2D(2,2))

    mdl.add(Flatten())

    mdl.add(Dense(512, activation='relu'))

    mdl.add(Dropout(0.25))

    mdl.add(
        tfpl.DenseReparameterization(
            units=tfpl.OneHotCategorical.params_size(6), 
            activation=None,
            kernel_prior_fn=tfpl.default_multivariate_normal_fn,
            kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            kernel_divergence_fn=divergence,
            bias_prior_fn=tfpl.default_multivariate_normal_fn,
            bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            bias_divergence_fn=divergence
            )
        )

    mdl.add(tfpl.OneHotCategorical(6))

    return mdl

mdl = model()

mdl.summary()

mdl.compile(
    loss=neg_loglike,
    optimizer=Adam(learning_rate=0.005),
    metrics=['accuracy'],
    experimental_run_tf_function=False
    )

history_mdl = mdl.fit(
    trgen,
    epochs=100,
    verbose=1
    )

try:
    history_mdl.save("/Users/karljaehnig/Repositories/bayesian_NN_experiments/bcnn_classifier/saved_models/intel_model")
except:
    mdl.save("/Users/karljaehnig/Repositories/bayesian_NN_experiments/bcnn_classifier/saved_models/intel_model")






