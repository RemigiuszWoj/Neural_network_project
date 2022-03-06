from ast import Pass
from re import M
import sys
from matplotlib import pyplot
import os
import ssl

from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow.keras.optimizers import SGD, Adam
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator


LAYER_NUMBER = [1, 2, 3]
NUMBERS_OF_NEURON = [32, 64, 128]

OPTIMIZER = ["SGD", "SGD_MOMENTUM", "ADAM"]
LEARNING_RATE = [0.001]
BETA_1 = [0.9]
BETA_2 = [0.999]
MOMENTUM = [0.9]

DATA_AUGMENTATION_ENEABLE_SWITCH = [False, True]
FIT_MODEL_EPOCH = [5, 50, 100]
FIT_MODEL_BATCH_SIZE = [64]

DROPOUT_ENEABLE = [False, True]
DROPOUT_VALUE = [0.2, 0.5, 0.8 ]


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

ssl._create_default_https_context = ssl._create_unverified_context

# load train and test dataset
def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY


# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm

def add_dropout(model, value:float=0.2, eneable:bool=False):
    if eneable:
        model.add(Dropout(value))



def layer_input(model, layer_number:int=1, dropout_value:float=0.2, dropout_eneable:bool=False) -> None:
    """
    Specify VGG blocks
    """
    if layer_number == 1:
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        add_dropout(model=model,value=dropout_value, eneable=dropout_eneable)
    elif layer_number == 2:
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        add_dropout(model=model,value=dropout_value, eneable=dropout_eneable)
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        add_dropout(model=model,value=dropout_value, eneable=dropout_eneable)
    elif layer_number == 3:
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        add_dropout(model=model,value=dropout_value, eneable=dropout_eneable)
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        add_dropout(model=model,value=dropout_value, eneable=dropout_eneable)
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        add_dropout(model=model,value=dropout_value, eneable=dropout_eneable)


def specify_number_of_neuron(model, numers_of_neuron:int=128, dropout_value:float=0.2, dropout_eneable:bool=False):
    """
    Specifi number of neuron
    """
    model.add(Dense(numers_of_neuron, activation='relu', kernel_initializer='he_uniform'))
    add_dropout(model=model,value=dropout_value, eneable=dropout_eneable)
    model.add(Dense(10, activation='softmax'))


def select_optimizer(optimizer:str = "SGD_MOMENTUM",
                     learning_rate:float=0.001,
                     beta_1:float = 0.9,
                     beta_2:float=0.999,
                     momentum:float=0.9,):
    if optimizer == "ADAM":
        current_optymizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
    elif optimizer == "SGD_MOMENTUM":
        current_optymizer = SGD(learning_rate=learning_rate, momentum=momentum)
    elif optimizer == "SGD":
        current_optymizer = SGD(learning_rate=learning_rate)
    return current_optymizer

# define cnn model
def define_model(layer_number:int=1,
                 numers_of_neuron:int=128,
                 dropout_value:float=0.2,
                 dropout_eneable:bool=False,
                 optimizer:str = "SGD_MOMENTUM",
                 learning_rate:float=0.001,
                 beta_1:float = 0.9,
                 beta_2:float=0.999,
                 momentum:float=0.9
                 ):
    model = Sequential()
    layer_input(model=model,layer_number=layer_number, dropout_value=dropout_value, dropout_eneable=dropout_eneable)
    model.add(Flatten())
    specify_number_of_neuron(model=model,
                             numers_of_neuron=numers_of_neuron,
                             dropout_value=dropout_value,
                             dropout_eneable=dropout_eneable)
    # model.add(Dense(10, activation='softmax'))
    # compile model
    # # opt = SGD(learning_rate=0.001, momentum=0.9)
    opt = select_optimizer(optimizer=optimizer, learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, momentum=momentum)
    
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model



# define cnn model
def define_model_old(neuron:int):
    model = Sequential()
    # if neuron == 1:
    #     model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    #     model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    #     model.add(MaxPooling2D((2, 2)))
    if neuron == 1:
        pass
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()


def data_augmentation_eneable(model, trainX, trainY, testX, testY,
                             fit_model_epoch:int=5,
                             fit_model_batch_size:int=64,
                             eneable:bool=False):
    if eneable == True:
        # create data generato
        datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
        # prepare iterator
        it_train = datagen.flow(trainX, trainY, batch_size=fit_model_batch_size)
         # fit model
        steps = int(trainX.shape[0] / 64)
        history = model.fit_generator(it_train, steps_per_epoch=steps, epochs=fit_model_epoch, validation_data=(testX, testY))
	# evaluate model

    else:
        # fit model
        history = model.fit(trainX, trainY, epochs=fit_model_epoch, batch_size=fit_model_batch_size, validation_data=(testX, testY))
    return history

# run the test harness for evaluating a model
def run_test_harness(layer_number:int=1,
                     numers_of_neuron:int=128,
                     dropout_value:float=0.2,
                     dropout_eneable:bool=False,
                     optimizer:str = "SGD_MOMENTUM",
                     learning_rate:float=0.001,
                     beta_1:float = 0.9,
                     beta_2:float=0.999,
                     momentum:float=0.9,
                     fit_model_epoch:int=5,
                     fit_model_batch_size:int=64,
                     data_augmentation_eneable_switch:bool=False
                     ):
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # define model
    model = define_model(layer_number=layer_number,
                         numers_of_neuron=numers_of_neuron,
                         dropout_value=dropout_value,
                         dropout_eneable=dropout_eneable,
                         optimizer=optimizer,
                         learning_rate=learning_rate,
                         beta_1=beta_1,
                         beta_2=beta_2,
                         momentum=momentum)
    # fit model
    history = data_augmentation_eneable(model=model, trainX=trainX, trainY=trainY, testX=testX, testY=testY,
                                        fit_model_epoch=fit_model_epoch,
                                        fit_model_batch_size=fit_model_batch_size,
                                        eneable=data_augmentation_eneable_switch)
    # evaluate model
    _, acc = model.evaluate(testX, testY,)
    print('> %.3f' % (acc * 100.0))
    # learning curves
    summarize_diagnostics(history)


def detonate():
    for layer_number in LAYER_NUMBER:
            for numbers_of_neuron in NUMBERS_OF_NEURON:
                for optimizer in OPTIMIZER:
                    for learnin_rate in LEARNING_RATE:
                        for beta_1 in BETA_1:
                            for beta_2 in BETA_2:
                                for momentum in MOMENTUM:
                                    for data_augmentation_eneable_switch in DATA_AUGMENTATION_ENEABLE_SWITCH:
                                        for fit_model_epoch in FIT_MODEL_EPOCH:
                                            for fit_model_batch_size in FIT_MODEL_BATCH_SIZE:
                                                for dropout_eneable in DROPOUT_ENEABLE:
                                                    for dropout_value in DROPOUT_VALUE:

                                                        run_test_harness(layer_number=layer_number,
                                                                numers_of_neuron=numbers_of_neuron,
                                                                dropout_value=dropout_value,
                                                                dropout_eneable=dropout_eneable,
                                                                optimizer=optimizer,
                                                                learning_rate=learnin_rate,
                                                                beta_1=beta_1,
                                                                beta_2=beta_2,
                                                                momentum=momentum,
                                                                fit_model_epoch=fit_model_epoch,
                                                                fit_model_batch_size=fit_model_batch_size,
                                                                data_augmentation_eneable_switch=data_augmentation_eneable_switch)


# TO DO
# zapisywanie wynikiow
# automatyzacja
# tablica pomylek + dokładność
# rozmiar sieci(liczba warst + neurony), dropout + wspolczynnik, augmentacja, metody SGD, SGD z momentem ADAM


# entry point, run the test harness
# print("cifar10")
# run_test_harness(layer_number=1,
#                  numers_of_neuron=128,
#                  dropout_value=0.2,
#                  dropout_eneable=False,
#                  optimizer="SGD_MOMENTUM",
#                  learning_rate=0.001,
#                  beta_1=0.9,
#                  beta_2=0.999,
#                  momentum=0.9,
#                  fit_model_epoch=5,
#                  fit_model_batch_size=64,
#                  data_augmentation_eneable_switch=False)
