'''
Created on Sep 30, 2017

@author: mroch
'''

import numpy as np

from keras.callbacks import Callback
from keras.layers import Dense, Input
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras import metrics

from sklearn.model_selection import StratifiedKFold

from mydsp.utils import Timer

class ErrorHistory(Callback):
    def on_train_begin(self, logs={}):
        self.error = []

    def on_batch_end(self, batch, logs={}):
        self.error.append(100 - logs.get('acc'))


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def feed_forward_model(specification):
    """feed_forward_model - specification list
    Create a feed forward model given a specification list
    Each element of the list represents a layer and is formed by a tuple.
    
    (layer_constructor, 
     positional_parameter_list,
     keyword_parameter_dictionary)
    
    Example, create M dimensional input to a 3 layer network with 
    20 unit ReLU hidden layers and N unit softmax output layer
    
    [(Dense, [20], {'activation':'relu', 'input_dim': M}),
     (Dense, [20], {'activation':'relu', 'input_dim':20}),
     (Dense, [N], {'activation':'softmax', 'input_dim':20})
    ]

    """
    model = Sequential()
    
    for item in specification:
        layertype = item[0]
        # Construct layer and add to model
        # This uses Python's *args and **kwargs constructs
        #
        # In a function call, *args passes each item of a list to 
        # the function as a positional parameter
        #
        # **args passes each item of a dictionary as a keyword argument
        # use the dictionary key as the argument name and the dictionary
        # value as the parameter value
        #
        # Note that *args and **args can be used in function declarations
        # to accept variable length arguments.
        layer = layertype(*item[1], **item[2])
        model.add(layer)
        
    return model
        
class CrossValidator:
    debug = False
    
    def __init__(self, Examples, Labels, model_spec, n_folds=10, epochs=100):
        """CrossValidator(Examples, Labels, model_spec, n_folds, epochs)
        Given a list of training examples in Examples and a corresponding
        set of class labels in Labels, train and evaluate a learner
        using cross validation.
        
        arguments:
        Examples:  feature matrix, each row is a feature vector
        Labels:  Class labels, one per feature vector.  Class labels
            can be strings.
        n_folds:  Number of folds in experiment
        epochs:  Number of times through data set
        model_spec: Specification of model to learn, see 
            feed_forward_model() for details and example  
        
        
        """     
        
        # This function should iterate over the K splits
        # calling train_and_evaluate_model for each split.
        # The error for the fold, the model, and the loss will be returned
        # and these should be retained
        #
        # Read about scikit's StratifiedKFold to learn how to run
        # the splits.
        skf = StratifiedKFold(n_folds)
        # creating a 2D array to hold error rates
        error_rate = np.zeros([n_folds,1])
        # creating a list of models
        model = []
        # loss history from model.fit()
        loss = []
        idx = 0
        # get train and test sizes for each fold
        for train_idx,test_idx in skf.split(Examples,Labels):
            error_rate[idx], m, l = self.train_and_evaluate__model(Examples,Labels,train_idx,test_idx,model_spec)
            print("Fold #:",idx)
            print("Error Rate:",error_rate[idx])
            idx += 1
            model.append(m)
            loss.append(l)

        self.errors = error_rate
        self.models = model
        self.losses = loss

    def train_and_evaluate__model(self, examples, labels, train_idx, test_idx, 
                                  model_spec, batch_size=100, epochs=100):
        """train_and_evaluate__model(examples, labels, train_idx, test_idx,
                model_spec, batch_size, epochs)
                
        Given:
            examples - List of examples in column major order
                (# of rows is feature dim)
            labels - list of corresponding labels
            train_idx - list of indices of examples and labels to be learned
            test_idx - list of indices of examples and labels of which
                the system should be tested.
            model_spec - Model specification, see feed_forward_model
                for details and example
        Optional arguments
            batch_size - size of minibatch
            epochs - # of epochs to compute
            
        Returns error rate, model, and loss history over training
        """
    
        # Useful functions
        # np_utils.to_categorical
        #
        # ErrorHistory() and LossHistory() are callback classes
        # that can be used when calling the fit function.  Only
        # the LossHistory is needed for this assignment.
        #
        # Don't forget to convert accuracy to error

        # Nx1 vector of our 3 categories -- Lecture notes
        label_vector = np_utils.to_categorical(labels)
        loss_h_obj = LossHistory()

        # model constructed using the feed_forward_model()
        model = feed_forward_model(model_spec)
        # defining model using optimizer, loss and metric --> lecture slides
        model.compile(optimizer="Adam",loss = "categorical_crossentropy", metrics = [metrics.categorical_accuracy])
        # if debug is set to True print the model summary
        if CrossValidator.debug:
            model.summary() # prints model architecture

        # training the model using given epoch and mini-batch size; default : 10, 100 respectively
        model.fit(examples[train_idx],label_vector[train_idx],batch_size=batch_size,epochs=epochs,callbacks=[loss_h_obj],verbose=CrossValidator.debug)
        # evaluating the performance based on the trained data
        eval_results = model.evaluate(examples[test_idx],label_vector[test_idx],verbose=CrossValidator.debug)
        # returning the error rate , model and loss history
        print("accuracy:",eval_results[1])
        return [1-eval_results[1], model,loss_h_obj]
      
    def get_models(self):
        "get_models() - Return list of models created by cross validation"
        return self.models
    
    def get_errors(self):
        "get_errors - Return list of error rates from each fold"
        return self.errors
    
    def get_losses(self):
        "get_losses - Return list of loss histories associated with each model"
        return self.losses
          
        
        


    