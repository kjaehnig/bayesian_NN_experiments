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

    - We can also split these distributions by digit. Even in the correctly classified samples, there is a spread in the entropy of those predictions, indicating that the model has higher uncertainty about these classifications. The incorrectly classified samples can also illustrate where the model might be sure of what turns out to be an incorrect prediction. Lower uncertainty on an incorrect classification can help identify if the model is over-fitting to the data.
    ![classification entropy distributions](https://github.com/kjaehnig/bayesian_NN_experiments/blob/main/bdenseNN_classifier/bdnn_per_class_classification_entropy.png)

    