'''
Created on Dec 2, 2017

@author: mroch
'''

from keras.callbacks import Callback

class ErrorHistory(Callback):
    def on_train_begin(self, logs={}):
        self.error = []

    def on_batch_end(self, batch, logs={}):
        self.error.append(100 - logs.get('categorical_accuracy'))


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
