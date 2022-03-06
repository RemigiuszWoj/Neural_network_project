from ast import Pass
from cProfile import run
from re import M
import sys
from typing import Counter
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
# from sklern.metrics import  


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

DATA_NAME = ["cifar10"]

RUN = [1, 2, 3]
COUNTER = [0]


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
    opt = select_optimizer(optimizer=optimizer, learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, momentum=momentum)
    
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model



# define cnn model
def define_model_old(neuron:int):
    model = Sequential()
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
def summarize_diagnostics(history, counter:int=0 , data_name:str = "cifar10", run:int=0):
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
    # filename = sys.argv[0].split('/')[-1]
    # filename ="cifar10" + os.sep + "plot" + os.sep + data_name + "_" + str(counter)
    filename =data_name + os.sep + str(run) + os.sep + "plot" + os.sep + data_name + "_" + str(counter)


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


def create_log(layer_number:int=1,
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
               data_augmentation_eneable_switch:bool=False,
               counter:int = 0,
               accuracy:float = 0.0,
               data_name:str = "cifar10",
               run:int=0,
               ):
    my_data_file = open(data_name + os.sep + str(run) + os.sep + "data_log" + os.sep + data_name + "_" + str(counter) + ".txt","w+")
    my_data_file.write("layer_number:" + str(layer_number) + "\n")
    my_data_file.write("numers_of_neuron:" + str(numers_of_neuron) + "\n")
    my_data_file.write("dropout_value:" + str(dropout_value) + "\n")
    my_data_file.write("dropout_eneable:" + str(dropout_eneable) + "\n")
    my_data_file.write("optimizer:" + str(optimizer) + "\n")
    my_data_file.write("learning_rate:" + str(learning_rate) + "\n")
    my_data_file.write("beta_1:" + str(beta_1) + "\n")
    my_data_file.write("beta_2:" + str(beta_2) + "\n")
    my_data_file.write("momentum:" + str(momentum) + "\n")
    my_data_file.write("fit_model_epoch:" + str(fit_model_epoch) + "\n")
    my_data_file.write("fit_model_batch_size:" + str(fit_model_batch_size) + "\n")
    my_data_file.write("data_augmentation_eneable_switch:" + str(data_augmentation_eneable_switch) + "\n")
    my_data_file.write("accuracy:" + str(accuracy) + "\n")
    my_data_file.write("data_name:" + str(data_name) + "\n")
    my_data_file.write("run:" + str(run) + "\n")
    my_data_file.write("counter:" + str(counter) + "\n")

    my_data_file.close()
    


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
                     fit_model_epoch:int=100,
                     fit_model_batch_size:int=64,
                     data_augmentation_eneable_switch:bool=False,
                     counter:int = 0,
                     run:int = 0,
                     data_name = DATA_NAME[0]
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
    # print('> %.3f' % (acc * 100.0))
    # create confiusion matrix
    matrix = confusion_matrix(testY, testX)
    print(matrix)

    create_log(layer_number=layer_number,
               numers_of_neuron=numers_of_neuron,
               dropout_value=dropout_value,
               dropout_eneable=dropout_eneable,
               optimizer=optimizer,
               learning_rate=learning_rate,
               beta_1=beta_1,
               beta_2=beta_2,
               momentum=momentum,
               fit_model_epoch=fit_model_epoch,
               fit_model_batch_size=fit_model_batch_size,
               data_augmentation_eneable_switch=data_augmentation_eneable_switch,
               counter=counter,
               accuracy=acc,
               data_name=data_name,
               run=run
              )
    # learning curves
    summarize_diagnostics(history, counter=counter, data_name=data_name, run=run)



def detonate():
    for run in RUN:
        counter = 0
        data_name = DATA_NAME[0]
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

                                                            counter = counter + 1
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
                                                                    data_augmentation_eneable_switch=data_augmentation_eneable_switch,
                                                                    counter = counter,
                                                                    run=run,
                                                                    data_name=data_name
                                                                    )

                                                        
def zad1_a(counter:int=0, fit_model_epoch:int=5,data_name:str=DATA_NAME[0]):
    """
    layer_numer
    numbers_of_neuron
    """

    for run in RUN:
        for layer_number in LAYER_NUMBER:
            for numers_of_neuron in NUMBERS_OF_NEURON:
                counter = counter + 1
                run_test_harness(layer_number=layer_number,
                                 numers_of_neuron=numers_of_neuron,
                                 run=run,
                                 fit_model_epoch=fit_model_epoch,
                                 data_name=data_name,
                                 counter=counter
                                )


def zad1_b(counter:int=0, fit_model_epoch:int=5,data_name:str=DATA_NAME[0]):
    """
    dropout
    """

    for run in RUN:
        for dropout_eneable in DROPOUT_ENEABLE:
            for dropout_value in DROPOUT_VALUE:
                counter = counter + 1
                run_test_harness(dropout_value=dropout_value,
                                 dropout_eneable=dropout_eneable,
                                 run=run,
                                 data_name=data_name,
                                 fit_model_epoch=fit_model_epoch,
                                 counter=counter,
                                 )



def zad1_c(counter:int=0, fit_model_epoch:int=5,data_name:str=DATA_NAME[0]):
    """
    dropout
    """

    for run in RUN:
        for data_augmentation_eneable_switch in DATA_AUGMENTATION_ENEABLE_SWITCH:
            counter = counter + 1
            run_test_harness(data_augmentation_eneable_switch=data_augmentation_eneable_switch,
                             run=run,
                             data_name=data_name,
                             fit_model_epoch=fit_model_epoch,
                             counter=counter,
                            )


def zad1_d(counter:int=0, fit_model_epoch:int=5,data_name:str=DATA_NAME[0]):
    """
    optimizer
    """

    for run in RUN:
        for optimizer in OPTIMIZER:
            counter = counter + 1
            run_test_harness(optimizer=optimizer,
                             run=run,
                             data_name=data_name,
                             fit_model_epoch=fit_model_epoch,
                             counter=counter,
                            )


def zad1_reference(counter:int=0, fit_model_epoch:int=5,data_name:str=DATA_NAME[0]):
    """
    reference
    """

    for run in RUN:
        counter = counter + 1
        run_test_harness(run=run,
                         data_name=data_name,
                         fit_model_epoch=fit_model_epoch,
                         counter=counter,
                        )


def zad1_test_run(counter:int=0, fit_model_epoch:int=5,data_name:str=DATA_NAME[0]):
    """
    test
    """
    counter = counter + 1
    run_test_harness(data_name=data_name,
                     fit_model_epoch=fit_model_epoch,
                     counter=counter,
                    )


def zad1_all(counter:int=0, fit_model_epoch:int=5,data_name:str=DATA_NAME[0]):
    zad1_a(counter=counter,fit_model_epoch=fit_model_epoch,data_name=data_name)
    zad1_b(counter=counter,fit_model_epoch=fit_model_epoch,data_name=data_name)
    zad1_c(counter=counter,fit_model_epoch=fit_model_epoch,data_name=data_name)
    zad1_d(counter=counter,fit_model_epoch=fit_model_epoch,data_name=data_name)
    zad1_reference(counter=counter,fit_model_epoch=fit_model_epoch,data_name=data_name)


# TO DO
# tablica pomylek 

# entry point, run the test harness
print(DATA_NAME[0])
# zad1_all(counter=COUNTER[0],fit_model_epoch=1,data_name=DATA_NAME[0])
zad1_test_run(fit_model_epoch=1)
