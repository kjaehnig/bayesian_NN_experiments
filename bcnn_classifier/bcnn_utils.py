from tensorflow.keras.preprocessing.image import ImageDataGenerator

def img_gen(trdir, tsdir):

    trdata = ImageDataGenerator(rescale=1/255)
    tsdata = ImageDataGenerator(rescale=1/255)


    trgen = trdata.flow_from_directory(
        trdir,
        target_size=(150,150),
        batch_size=100,
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


