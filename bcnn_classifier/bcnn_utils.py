from tensorflow.keras.preprocessing.image import ImageDataGenerator

def img_gen(trdir, tsdir):

    trdata = ImageDataGenerator(rescale=1/255,
                                horizontal_flip=True,
                                rotation_range=0)
    tsdata = ImageDataGenerator(rescale=1/255)


    trgen = trdata.flow_from_directory(
        trdir,
        target_size=(150,150),
        batch_size=50,
        class_mode='categorical',
        subset='training'
        )

    tsgen = tsdata.flow_from_directory(
        tsdir,
        target_size=(150,150),
        batch_size=50,
        class_mode='categorical'
        )

    return trgen, tsgen


def display_examples(imggen):
    """
        Display 25 images from the images array with its corresponding labels
    """
    class_labels = list(imggen.class_indices)
    imggen.batch_size=50
    randbatch = next(imggen)

    fig = plt.figure(figsize=(10,10))
    fig.suptitle("Some examples of images of the dataset", fontsize=16)
    for ii in range(25):
        image = randbatch[0][ii]
        true_label = np.argmax(randbatch[1][ii])

        plt.subplot(5,5,ii+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)
        plt.xlabel(class_labels[true_label])
    plt.show()


def plot_confusion_matrix(cm, class_names):
  """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
  import matplotlib.pyplot as plt
  import itertools
  figure = plt.figure(figsize=(8, 8))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
  plt.title("Confusion matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

  # Compute the labels from the normalized confusion matrix.
  labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 4
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  return figure


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def log_confusion_matrix(epoch, logs):
  # Use the model to predict the values from the validation dataset.
  test_pred_raw = model.predict(tsgen)
  test_pred = np.argmax(test_pred_raw, axis=1)

  # Calculate the confusion matrix.
  cm = sklearn.metrics.confusion_matrix(tsgen.labels, test_pred)
  # Log the confusion matrix as an image summary.
  figure = plot_confusion_matrix(cm, class_names=class_names)
  cm_image = plot_to_image(figure)

  # Log the confusion matrix as an image summary.
  with file_writer_cm.as_default():
    tf.summary.image("epoch_confusion_matrix", cm_image, step=epoch)


def grab_predict_random_image(bmodel,imggen):


    n_iter = 300
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
    # img_resize = (cv2.resize(img, dsize=(150, 150), interpolation=cv2.INTER_CUBIC))/255.
    
    predicted_probabilities = np.empty(shape=(n_iter, num_classes))
    
    for i in range(n_iter):
        predicted_probabilities[i] = bmodel(img[np.newaxis,:]).mean().numpy()[0]
    # print(predicted_probabilities)
    pct_2p5 = np.array([np.percentile(predicted_probabilities[:, i], 2.5) for i in range(num_classes)])
    pct_97p5 = np.array([np.percentile(predicted_probabilities[:, i], 97.5) for i in range(num_classes)])
    
    pct_50 = np.array([np.percentile(predicted_probabilities[:,i], 50) for i in range(num_classes)])
    # mdl_pred = bmodel.predict(img)
    # print(mdl_pred)
    # fig, ax = plt.subplots(figsize=(12, 6))
    bar = barax.bar(np.arange(num_classes), pct_97p5, color='red')
    bar[true_label].set_color('green')
    barax.bar(np.arange(num_classes), pct_2p5-0.02, lw=2, color='white')
    barax.set_xticklabels([''] + [x for x in class_labels])
    barax.set_ylim([0, 1])
    barax.set_ylabel('Probability')
    barax.set_title(f'50p model pred: {class_labels[np.argmax(pct_50)]}')

    plt.show()





# Function to define the spike and slab distribution

def spike_and_slab(event_shape, dtype):
    distribution = tfd.Mixture(
        cat=tfd.Categorical(probs=[0.5, 0.5]),
        components=[
            tfd.Independent(tfd.Normal(
                loc=tf.zeros(event_shape, dtype=dtype), 
                scale=1.0*tf.ones(event_shape, dtype=dtype)),
                            reinterpreted_batch_ndims=1),
            tfd.Independent(tfd.Normal(
                loc=tf.zeros(event_shape, dtype=dtype), 
                scale=10.0*tf.ones(event_shape, dtype=dtype)),
                            reinterpreted_batch_ndims=1)],
    name='spike_and_slab')
    return distribution

def get_convolutional_reparameterization_layer(input_shape=None, divergence_fn=None, filters=16,name=None):
    """
    This function should create an instance of a Convolution2DReparameterization 
    layer according to the above specification. 
    The function takes the input_shape and divergence_fn as arguments, which should 
    be used to define the layer.
    Your function should then return the layer instance.
    """
    if input_shape == None:
        layer = tfpl.Convolution2DReparameterization(
            filters=filters, kernel_size=(3, 3),
            activation='relu', padding='same',
            kernel_prior_fn=tfpl.default_multivariate_normal_fn,
            kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            kernel_divergence_fn=divergence,
            bias_prior_fn=tfpl.default_multivariate_normal_fn,
            bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            bias_divergence_fn=divergence,
            name=name
            )
    if input_shape != None:
        layer = tfpl.Convolution2DReparameterization(
            input_shape=input_shape, filters=32, kernel_size=(3, 3),
            activation='relu', padding='same',
            kernel_prior_fn=tfpl.default_multivariate_normal_fn,
            kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            kernel_divergence_fn=divergence,
            bias_prior_fn=tfpl.default_multivariate_normal_fn,
            bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            bias_divergence_fn=divergence,
            name=name
            )
    return layer


def get_prior(kernel_size, bias_size, dtype=None):
    """
    This function should create the prior distribution, consisting of the 
    "spike and slab" distribution that is described above. 
    The distribution should be created using the kernel_size, bias_size and dtype
    function arguments above.
    The function should then return a callable, that returns the prior distribution.
    """
    n = kernel_size+bias_size  
    prior_model = Sequential([tfpl.DistributionLambda(lambda t : spike_and_slab(n, dtype))])
    return prior_model


def get_posterior(kernel_size, bias_size, dtype=None):
    """
    This function should create the posterior distribution as specified above.
    The distribution should be created using the kernel_size, bias_size and dtype
    function arguments above.
    The function should then return a callable, that returns the posterior distribution.
    """
    n = kernel_size + bias_size
    return Sequential([
        tfpl.VariableLayer(tfpl.IndependentNormal.params_size(n), dtype=dtype),
        tfpl.IndependentNormal(n)
    ])


def get_dense_variational_layer(prior_fn, posterior_fn, kl_weight):
    """
    This function should create an instance of a DenseVariational layer according 
    to the above specification. 
    The function takes the prior_fn, posterior_fn and kl_weight as arguments, which should 
    be used to define the layer.
    Your function should then return the layer instance.
    """
    return tfpl.DenseVariational(
        units=6, make_posterior_fn=posterior_fn, make_prior_fn=prior_fn, kl_weight=kl_weight
        )

def get_dense_reparameterization_layer(divergence_fn=None, name=None):

    """

    """

    layer = tfpl.DenseReparameterization(
        units=tfpl.OneHotCategorical.params_size(6),
        activation=None,
        kernel_prior_fn=tfpl.default_multivariate_normal_fn,
        kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
        kernel_divergence_fn=divergence_fn,
        bias_prior_fn=tfpl.default_multivariate_normal_fn,
        bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
        bias_divergence_fn=divergence_fn,
        name=name
        )

    return layer


def generate_labels_and_preds(mdl, tsgen):
    """
    This function corrects not using a seed with the Image Data Generators
    when trying to compute validation metrics or confusion matricies. It 
    simply computes predictions from generator batches, as well as the batch
    predictions, returning one-to-one arrays of truths and predictions.
    """
    from tqdm import tqdm
    orig_bs = tsgen.batch_size
    tsgen.batch_size = 100
    lbls = np.zeros(tsgen.samples)+999
    prds = np.zeros(tsgen.samples)+999
    for ii in tqdm(range(int(tsgen.samples/tsgen.batch_size))):
         btch = next(tsgen)
         blbl = np.argmax(btch[1],axis=1)
         bprd = np.argmax(mdl.predict(btch[0],verbose=0),axis=1)
         lbls[ii*tsgen.batch_size:ii*tsgen.batch_size+tsgen.batch_size] = blbl
         prds[ii*tsgen.batch_size:ii*tsgen.batch_size+tsgen.batch_size] = bprd
    tsgen.batch_size = orig_bs
    return lbls, prds


def get_correct_indices(model, x, labels):
    y_model = model(x)
    correct = np.argmax(y_model.mean(), axis=1) == np.squeeze(labels)
    correct_indices = [i for i in range(x.shape[0]) if correct[i]]
    incorrect_indices = [i for i in range(x.shape[0]) if not correct[i]]
    return correct_indices, incorrect_indices



def plot_entropy_distribution(model, x, labels):
    probs = model(x).mean().numpy()
    entropy = -np.sum(probs * np.log(probs), axis=1)

    fig,axes = plt.subplots(1,2, figsize=(10,4))

    for i,category in zip(range(2),['correct','incorrect']):
        entropy_category = entropy[get_correct_indices(model,x,labels)[i]]
        mean_entropy = np.mean(entropy_category)
        num_samples = entropy_category.shape[0]
        title = f"{category}ly labeled ({num_samples/x.shape[0]*100:.1f})"
        axes[i].hist(entropy_category, weights=(1/num_samples)*np.ones(num_samples))
        axes[i].annotate(f"Mean: {mean_entropy:.3f} bits", (0.4,0.9), ha='center')
        axes[i].set_xlabel('entropy (bits)')
        axes[i].set_ylim([0,1])
        axes[i].set_ylabel('probability')
        axes[i].set_title(title)


    plt.show()












