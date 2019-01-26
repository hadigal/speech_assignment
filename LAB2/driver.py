'''
Created on Nov 10, 2018

@author: mroch
'''

import os

# Add-on modules
from keras.layers import Dense, Dropout, LSTM, Masking, TimeDistributed, GRU
from keras import regularizers

import numpy as np

import matplotlib.pyplot as plt

from dsp.features import Features
from timit.corpus import Corpus

from myclassifier.buildmodels import build_model
from myclassifier.crossvalidator import CrossValidator
import myclassifier.recurrent
import myclassifier.feedforward
from keras.layers.normalization import BatchNormalization
from myclassifier.batchgenerator import PaddedBatchGenerator

def main():

    adv_ms = 10     # frame advance and length
    len_ms = 20

    # Adjust to your system
    TimitBaseDir = 'C:\Hrishi_documents\SDSU\FALL_2018\CS_682\lab2\\timit-for-students'
    corpus = Corpus(TimitBaseDir, os.path.join(TimitBaseDir, 'wav'))
    print("CORPUS PATH:{}".format(os.path.join(TimitBaseDir, 'wav')))
    phonemes = corpus.get_phonemes()  # List of phonemes
    phonemesN = len(phonemes)  # Number of categories

    # Get utterance keys
    devel = corpus.get_utterances('train')  # development corpus
    eval = corpus.get_utterances('test')  # evaluation corpus


    features = Features(adv_ms, len_ms, corpus.get_audio_dir())
    # set features storage location
    features.set_cacheroot(os.path.join(TimitBaseDir, 'feature_cache'))
    corpus.set_feature_extractor(features)

    # Example of retrieving features; also allows us to determine
    # the dimensionality of the feature vector
    f = corpus.get_features(devel[0])
    # Determine input shape
    input_dim = f.shape[1]

    # Check if any features have Inf/NaN in them
    data_sanity_check = False
    if data_sanity_check:
        idx = 0
        for utterances in [devel, eval]:
            for u in utterances:
                f = corpus.get_features(u)
                # Check for NaN +/- Inf
                nans = np.argwhere(np.isnan(f))
                infs = np.argwhere(np.isinf(f))
                if len(nans) > 0 or len(infs) > 0:
                    print(u)
                    if len(nans) > 0:
                        print("NaN")
                        print(nans)

                    if len(infs) > 0:
                        print('Inf')
                        print(infs)
                    pass    # Good place for a breakpoint...

                idx = idx + 1
                if idx % 100 == 0:
                    print("idx %d"%(idx))


    # Most of your work in the driver is specifying models, summarizing
    # results, designing a search strategy...

    ###### model with mixed regularization ######

    # models_rnn = [
    #     lambda dim, width, dropout, l2:
    #     [(Masking, [], {"mask_value": 0.,
    #                     "input_shape": [None, dim]}),
    #      (LSTM, [width], {
    #          "return_sequences": True,
    #          "kernel_regularizer": regularizers.l1(l1),
    #          "recurrent_regularizer": regularizers.l1(l1)
    #      }),
    #      (Dropout, [dropout], {}),
    #      (LSTM, [width], {
    #          "return_sequences": True,
    #          "kernel_regularizer": regularizers.l1(l1),
    #          "recurrent_regularizer": regularizers.l1(l1)
    #      }),
    #      (Dropout, [dropout], {}),
    #      (LSTM, [width], {
    #       "return_sequences":True,
    #       "kernel_regularizer":regularizers.l2(l2),
    #       "recurrent_regularizer":regularizers.l2(l2)
    #      }),
    #      (Dropout, [dropout], {}),
    #      (BatchNormalization, [], {}),
    #      (Dense, [phonemesN], {'activation': 'softmax',
    #                            'kernel_regularizer': regularizers.l2(l2)},
    #       # The Dense layer is not recurrent, we need to wrap it in
    #       # a layer that that lets the network handle the fact that
    #       # our tensors have an additional dimension of time.
    #       (TimeDistributed, [], {}))
    #      ]
    # ]

    ####### model with best params #######

    models_rnn = [
        lambda dim, width, dropout, l2:
        [(Masking, [], {"mask_value": 0.,
                        "input_shape": [None, dim]}),
         (LSTM, [width], {
             "return_sequences": True,
             "kernel_regularizer": regularizers.l2(l2),
             "recurrent_regularizer": regularizers.l2(l2)
         }),
         (Dropout, [dropout], {}),
         (LSTM, [width], {
             "return_sequences": True,
             "kernel_regularizer": regularizers.l2(l2),
             "recurrent_regularizer": regularizers.l2(l2)
         }),
         (Dropout, [dropout], {}),
         (BatchNormalization, [], {}),
         (Dense, [phonemesN], {'activation': 'softmax',
                               'kernel_regularizer': regularizers.l2(l2)},
          # The Dense layer is not recurrent, we need to wrap it in
          # a layer that that lets the network handle the fact that
          # our tensors have an additional dimension of time.
          (TimeDistributed, [], {}))
         ]
    ]

    # rnn = build_model(models_rnn[0](input_dim, 45, .2, 0.001)) ----> 52%
    # rnn = build_model(models_rnn[0](input_dim, 45, .1, 0.01)) ----> 49%
    # rnn = build_model(models_rnn[0](input_dim, 30, .2, 0.01)) ----> 25%

    rnn = build_model(models_rnn[0](input_dim, 45, .2, 0.001))
    debug = True

    ########## uncomment to run the model with different params #############
    # HR: Running different models for with different settings to choose the
    # optimum one for the final run --> Grid search
    # width_param = [45,30] # HR: List to iterate over to test different node sizes
    # req_penalties = [0.01,0.002,0.001] # HR: List to iterate over different regularizations
    # do = [0.5,0.3,0.2] # HR: List to iterate over different dropouts

    # # list to hold the different models after build with various params
    # model_list = []
    # print("BUILDING MODELS..........\n")
    # for params_idx in range(len(width_param)):
    #     model_list.append([])
    #     for idx2 in range(len(do)):
    #         rnn_mod = build_model(models_rnn[0](input_dim,width_param[params_idx], do[idx2], req_penalties[idx2]))
    #         # cross validating the model with different params
    #         if debug:
    #             devel = devel[0:30]
    #         # Sample cross validation.  Note that we use smaller batch sizes
    #         # as each utterance has many exmaples (a sentence worth of phonemes)
    #         cv = CrossValidator(corpus, devel, rnn_mod,
    #                             myclassifier.recurrent.train_and_evaluate,
    #                             batch_size=10,
    #                             epochs=20,
    #                             n_folds=2)
    #         print("Errors for model[{}][{}] set of params:{}".format(params_idx,idx2,cv.get_errors()))
    #         model_list[params_idx].append(cv.get_models())
    # print("\nDONE BUILDING MODELS..........\n")

    # debug = True
    # ######### HR: This is for Final Run after finializing the params ###########
    if debug:
        devel = devel[0:30]
    # Sample cross validation.  Note that we use smaller batch sizes
    # as each utterance has many exmaples (a sentence worth of phonemes)
    cv = CrossValidator(corpus, devel, rnn,
                        myclassifier.recurrent.train_and_evaluate,
                        batch_size=100,
                        epochs=20,
                        n_folds=2)
    print("Errors:",cv.get_errors())
    print("\nDONE BUILDING MODELS..........\n")

    built_models = cv.get_models()

    ################## HR: EVALUATION LOGIC ########################
    # Evaluating the built model on the evaluation data to measure the accuracy
    # and error rate
    if debug:
        eval = eval[0:30]
    pbgObj = PaddedBatchGenerator(corpus=corpus, utterances=eval, batch_size=len(eval))
    (eval_features, eval_labels) = next(pbgObj)

    ########## This is for final run for model built with best params ########
    error_rate = 0
    tempErrorList = []
    c = 0
    for md in built_models:
        result = md.evaluate(eval_features, eval_labels, verbose=False)
        error_rate += (1 - result[1])
        tempErrorList.append((1 - result[1]))
        print("Accuracy for Model {}:{}\n".format(c, result[1]))
        c += 1

    print("Max error Rate:{}".format(max(tempErrorList)))
    print("Mean Error Rate:{}".format((sum(tempErrorList) / len(built_models))))

    ############# uncomment to run for different params #################
    # # HR: iterating over the list of models for evaluation with eval data
    # all_model_error_list = []
    # for idx1 in range(len(width_param)):
    #     all_model_error_list.append([])
    #     itr = 0
    #     for idx2 in range(len(do)):
    #         error_rate = 0
    #         tempErrorList = []
    #         c = 0
    #         for md in model_list[idx1][idx2]:
    #             result = md.evaluate(eval_features, eval_labels, verbose=False)
    #             error_rate += (1 - result[1])
    #             tempErrorList.append((1 - result[1]))
    #             print("Accuracy for Model{}{}{}:{}\n".format(idx1,idx2,c,result[1]))
    #             c += 1
    #         all_model_error_list.append(tempErrorList)
    #         print("Max error Rate for models of do and reg type{}:{}".format(itr,max(tempErrorList)))
    #         print("Mean Error Rate {}:{}".format(itr,(sum(tempErrorList) / len(model_list[idx1][idx2]))))
    #         itr += 1

if __name__ == '__main__':
    plt.ion()
    main()