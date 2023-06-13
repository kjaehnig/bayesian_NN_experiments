# bayesian_NN_experiments
Repository for various experiments with Bayesian neural networks in TensorFlow


## Quick list of projects

1. Bayesian convolutional neural network classifier using Intel images

    - Current best validation accuracy: ~83%

    - Bayesian NN can output probability intervals, illustrating how sure/unsure the model is of a classification.
    - A very sure classification of an image of a sea.
    ![sure sea classification](https://github.com/kjaehnig/bayesian_NN_experiments/blob/main/bcnn_classifier/bmdl2_sea_prediction_uncertainty.png)

    - An unsure classification of a different sea image.
    ![unsure sea classification](https://github.com/kjaehnig/bayesian_NN_experiments/blob/main/bcnn_classifier/bmdl2_sea_unsure_prediction_uncertainty.png)


2. Bayesian dense neural network classifier using MNIST 28x28 dataset with images flattened to vectors

    - Current best validation accuracy ~97%

    - The difference in entropy distributions of correct/incorrect classifications is much higher than with the bcnn above.
    ![classification entropy distributions](https://github.com/kjaehnig/bayesian_NN_experiments/blob/main/bdenseNN_classifier/bdnn_classification_entropy_plot.png)