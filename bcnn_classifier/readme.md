# cnn_classifiers
Storage directory for deep convolutional neural network classifier in keras trained on intel location images


## Classifier constructed in Tensorflow with Keras

#### Trained on Intel image classification dataset [downloadedable from Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification?resource=download)

![5x5 sample of Images](https://github.com/kjaehnig/bayesian_NN_experiments/blob/main/bcnn_classifier/twenty_five_intel_images_example.png)


#### Current BNN architecture drawn using keras 'plot_model' utility
![Latest model architecture](https://github.com/kjaehnig/bayesian_NN_experiments/blob/main/bcnn_classifier/Bmodel2.split.model.plot.png)


#### Confusion Matrix for best trained BNN (~83% accuracy)
![Current Confusion Matrix for best trained model (~83% accuracy)](https://github.com/kjaehnig/bayesian_NN_experiments/blob/main/bcnn_classifier/bmdl2_confusion_matrix.png)

#### Entropy Distributions for correctly/incorrectly classified images, illustrating how sure/unsure the BNN is about its classifications.
![Entropy distribution of BNN](https://github.com/kjaehnig/bayesian_NN_experiments/blob/main/bcnn_classifier/bmdl2_classification_entropy_distributions.png)

#### A BNN can produce probability intervals on classifications, illustrating how sure/unsure a classification is. Here is a very certain classification of the image of a sea.
![sea classification](https://github.com/kjaehnig/bayesian_NN_experiments/blob/main/bcnn_classifier/bmdl2_sea_prediction_uncertainty.png)

#### Here we see an uncertain classification of another image of a sea.
![sea classification](https://github.com/kjaehnig/bayesian_NN_experiments/blob/main/bcnn_classifier/bmdl2_sea_unsure_prediction_uncertainty.png)