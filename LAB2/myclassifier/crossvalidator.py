'''
Created on Dec 2, 2017

@author: mroch
'''

from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np

from dsp.utils import Timer
        
class CrossValidator:

    debug = False
    
    # If not None, N-fold cross validation will abort after processing N folds
    abort_after_N_folds = None
    
    def __init__(self, corpus, keys, model, model_train_eval,
                 n_folds=10, batch_size=100, epochs=100):
        """CrossValidator(corpus, keys, model_spec, n_folds, batch_size, epochs)
        Cross validation for sequence models
        Given a corpus object capable of retrieving features and a list of
        keys to access the data and labels from the corpus object, run a
        cross validation experiment
        
        
        arguments:
        corpus - Object representing data and labels.  Must support
          methods get_features and get_labels which both take one of the 
          keys passed in.  See timit.corpus.Corpus for an example of a class
          that supports this interface.
          get_features returns a feature matrix
          get_labels returns a list of start and stop indices as well as
              labels.  start[i], end[i], label[i] means that labe[i] is 
              present for features between indices of start[i] and end[i] 
              (inclusive).  
        keys - values that can be passed to corpus.get_feautures and
            corpus.get_labels to return data.  These are what will be split
            for the cross validation

        model: Keras model to learn
            e.g. result of buildmodels.build_model()
            
        model_train_evel: function that can be called to train and test
            a model.  Must conform to an interface that expects the following
            arguments:
                corpus - corpus object
                trainkeys - keys used to train
                testkeys - keys used to test
                model - keras network to be used
                batch_size - # examples to process per batch
                epochs - Number of passes through training data
                name - test name
        n_folds - # of cross validation folds
        batch_size - # examples to process per batch
        epochs - Number of passes through training data  
        """
        
        # Create a plan for k-fold testing with shuffling of examples
        kfold = KFold(n_folds, shuffle=True)
        
        # HINT:  As you are not working with actual samples here, but rather
        # utterance keys, create a list of indices into the utterances
        # list and use those when iterating over kfold.split()
        
        # Store the models errors, and losses in lists
        
        # It is also suggested that you might want to create a Timer
        # to track how long things  are (not that it will help things go
        # any faster)

        # HR: setting test name as fold number
        count = 'rnn'
        # HR: defining lists to hold model,error and loss after the model has finish training
        model_list = []
        err_list = []
        loss_list = []
        # init timer var for cal time elapsed
        time = Timer()

        # HR: iterate over the features and labels to create a list of errors, models and losses
        for (train_idx, test_idx) in kfold.split(keys):
            # HR: The function pointer points to the train_and_eval() from recurrent.py
            # this returns a tuple (err, model, loss)
            (err, model, loss) = model_train_eval(corpus,keys[train_idx],keys[test_idx],model,batch_size=batch_size,epochs = epochs)
            model_list.append(model)
            err_list.append(err)
            loss_list.append(loss)
            # count += 1

        # logging to time to know build time
        print("Time Elapsed:",time.elapsed())
        time.reset()
        self.errors = err_list
        self.models = model_list
        self.losses = loss_list
  
    def get_models(self):
        "get_models() - Return list of models created by cross validation"
        return self.models
    
    def get_errors(self):
        "get_errors - Return list of error rates from each fold"
        return self.errors
    
    def get_losses(self):
        "get_losses - Return list of loss histories associated with each model"
        return self.losses
