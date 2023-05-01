import os

import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec
from bcnn_utils import img_gen
import bcnn_utils as bcu


tfd = tfp.distributions
tfpl = tfp.layers


trimg_dir = "/Users/karljaehnig/archive/seg_train/seg_train/"
tsimg_dir = "/Users/karljaehnig/archive/seg_test/seg_test/"




def neg_loglike(ytrue, ypred):
    return -ypred.log_prob(ytrue)

def divergence(q,p,_):
    return tfd.kl_divergence(q,p)/14034


def bmodel():

    mdl = Sequential()
    mdl.add(
        tfpl.Convolution2DReparameterization(
            input_shape=(150,150,3),
            filters=8,
            kernel_size=(3,3),
            activation='relu',
            kernel_prior_fn=tfpl.default_multivariate_normal_fn,
            kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            kernel_divergence_fn=divergence,
            bias_prior_fn=tfpl.default_multivariate_normal_fn,
            bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            bias_divergence_fn=divergence
            )
        )
    mdl.add(MaxPooling2D(2,1))
    
    mdl.add(Conv2D(16, (3,3), activation='relu'))
    mdl.add(MaxPooling2D(2,2))

    mdl.add(Conv2D(32, (3,3), activation='relu'))
    mdl.add(MaxPooling2D(2,2))

    mdl.add(Conv2D(64, (3,3), activation='relu'))
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



def grab_predict_random_image(bmodel,imggen):

    #generate random image with imggen
    num_classes = imggen.num_classes
    batch_size = imggen.batch_size
    randint = np.random.randint(low=0, high=batch_size-1)
    randbatch = next(imggen)

    image = randbatch[0][randint]
    true_label = np.argmax(randbatch[1][randint])

    class_labels = list(imggen.class_indices)

    fig = plt.figure(figsize=(20,5))                                                           
    spec = GridSpec(nrows=25,ncols=100,figure=fig)

    imgax = fig.add_subplot(spec[:,:25])
    barax = fig.add_subplot(spec[:,30:])    
    #read image
    # img = cv2.imread(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    
    #show the image
    imgax.imshow(img)
    imgax.axis('off')
    imgax.set_title(f'actual label: {class_labels[true_label]}')
    img_resize = (cv2.resize(img, dsize=(150, 150), interpolation=cv2.INTER_CUBIC))/255.
    
    predicted_probabilities = np.empty(shape=(100, num_classes))
    
    for i in range(100):
        
        predicted_probabilities[i] = bmodel(img_resize[np.newaxis,...]).mean().numpy()[0]
    print(predicted_probabilities)
    pct_2p5 = np.array([np.percentile(predicted_probabilities[:, i], 16) for i in range(num_classes)])
    pct_97p5 = np.array([np.percentile(predicted_probabilities[:, i], 84) for i in range(num_classes)])
    
    pred50 = np.argmax(np.mean(predicted_probabilities))
    # fig, ax = plt.subplots(figsize=(12, 6))
    bar = barax.bar(np.arange(num_classes), pct_97p5, color='red')
    bar[true_label].set_color('green')
    bar = barax.bar(np.arange(num_classes), pct_2p5, 0.83, color='white')
    barax.set_xticklabels([''] + [x for x in class_labels])
    barax.set_ylim([0, 1])
    barax.set_ylabel('Probability')
    barax.set_title(f'median label: {class_labels[pred50]}')

    plt.show()



