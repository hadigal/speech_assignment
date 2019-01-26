import math
import time
import os.path

import numpy as np

import keras
from keras.callbacks import TensorBoard
from keras.utils import np_utils
from keras import metrics

import keras.backend as K
import tensorflow as tf
 
from dsp.utils import Timer

from .batchgenerator import PaddedBatchGenerator
from .histories import ErrorHistory, LossHistory
from .confusion import ConfusionTensorBoard, plot_confusion



def train_and_evaluate(corpus, train_utt, test_utt, 
                              model, batch_size=100, epochs=100, 
                              name="model"):
    """train_and_evaluate__model(corpus, train_utt, test_utt,
            model, batch_size, epochs)
            
    Given:
        corpus - Object for accessing labels and feature data
        train_utt - utterances used for training
        test_utt - utterances used for testing
        model - Keras model
    Optional arguments
        batch_size - size of minibatch
        epochs - # of epochs to compute
        name - model name
        
    Returns error rate, model, and loss history over training
    """

    # write me
    # HR: Compiling the model built in driver
    #compile(optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=[metrics.categorical_accuracy])
    print("Model:{} - Summary\n{}".format(name,model.summary()))

    # HR: creating a tensorboard object for visualizing results
    # var to fold logging directory
    file_name = 'C:\Hrishi_documents\SDSU\FALL_2018\CS_682\lab2\my_code\logs\{}'.format(int(time.time()))
    tb = TensorBoard(log_dir=file_name, histogram_freq=0, write_graph=True, write_grads=True)
    ctb = ConfusionTensorBoard(logdir=file_name, labels=corpus.get_phonemes(), writer = K.get_session())
    ctb.add_callbacks(model)

    # obj of Loss history and Error history for logging
    err_history = ErrorHistory()
    loss_history = LossHistory()

    ########## This is for debug purpose remove later ##############
    # HR: iterating over the list of utterances to create a list of labels and features
    # defination def __init__(self, corpus, utterances, batch_size=100):
    # train_ex = []
    # train_lab = []
    # for i in train_utt:
    #     train_ex.append(corpus.get_features(i))
    #     train_lab.append(corpus.get_labels(i))
    # test_ex = []
    # test_lab = []
    # for i in test_utt:
    #     test_ex.append(corpus.get_features(i))
    #     test_lab.append(corpus.get_labels(i))

    # HR: as mentioned in the problem set creating a obj of PaddedBatchGenerator
    testGen = PaddedBatchGenerator(corpus=corpus, utterances=test_utt, batch_size=batch_size)
    (test_feat,test_lab) = next(testGen)
    print("Extracted test features and labels.................\n")

    trainGen = PaddedBatchGenerator(corpus=corpus, utterances=train_utt, batch_size=batch_size)
    (train_feat, train_lab) = next(trainGen)
    print("Extracted train features and labels...............\n")

    # HR: if batch_size is fixed size
    model.fit_generator(trainGen, steps_per_epoch=trainGen.get_batches_per_epoch(),epochs=epochs, callbacks=[loss_history, tb,ctb],validation_data=[train_feat, train_lab])

    # HR: for test data and labels??
    # model.fit_generator(testGen, steps_per_epoch=testGen.get_batches_per_epoch(),epochs=epochs, callbacks=[loss_history, tb],validation_data=[test_feat, test_lab])

    # try using this for error due to batch norm layer
    # K.set_learning_phase(False)

    # logging all the losses and errors rates to terminal
    count = 0
    loss_l =[]
    for loss in loss_history.losses:
        # print("Loss[{}]:{}".format(count,loss))
        loss_l.append(loss)
        count += 1
    print("# of Loss: {}\nLosses History List: {}\n".format(count, loss_l))
    model_eval = model.evaluate(test_feat,test_lab,verbose=False)
    err = 1 - model_eval[1]
    print("Accuracy:{}\n".format(model_eval[1]))
    print("Error Rate:{}\n".format(err))
    return (err, model, loss_history)
