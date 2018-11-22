'''
Created on Oct 21, 2017

@author: mroch

'''

from mydsp.pca import PCA
from mydsp.utils import pca_analysis_of_spectra
from mydsp.utils import get_corpus, get_class, Timer
from mydsp.utils import fixed_len_spectrogram, spectrogram, plot_matrix

from mydsp.features import extract_features_from_corpus
from myclassifier.feedforward import CrossValidator

from keras.layers import Dense, Dropout
from keras import regularizers
from keras.utils import np_utils, plot_model

import numpy as np
import matplotlib.pyplot as plt

          
    
def main():
   
    
    files = get_corpus("/Users/macbook/Downloads/hrishi/tidigits-isolated-digits-wav/wav/train")
    files_train = files
    files_test = get_corpus("/Users/macbook/Downloads/hrishi/tidigits-isolated-digits-wav/wav/test")

    # for developing
    if False:
        truncate_to_N = 50
        print("Truncating t %d files"%(truncate_to_N))
        files[truncate_to_N:] = []  # truncate test for speed
        files_test[truncate_to_N:] = []  # truncate test for speed
        files_train[truncate_to_N:] = []  # truncate test for speed
    
    print("%d files"%(len(files)))
    
    adv_ms = 10
    len_ms = 20
    # We want to retain offset_s about the center
    offset_s = 0.25
    
    timer = Timer()

    # Executing timer.reset() shows the amount of time elapsed and resets the timer
    # hr: computing the pca transform on training data
    pca = pca_analysis_of_spectra(files_train,adv_ms,len_ms,offset_s)
    # HR: getting output categories for the training data set
    outputN = len(set(get_class(files_train)))

    # Specify model architectures
    model_list = lambda input_nodes :[
        # 3 layer 50x50xoutput baseline - No regularization wide
        [(Dense, [50], {'activation':'relu', 'input_dim':input_nodes.shape[1]}),
         (Dense, [50], {'activation':'relu', 'input_dim':50}),
         (Dense, [outputN], {'activation':'softmax', 'input_dim':50})
        ],
        # 5 layer deep net 20x20x20x20xoutput model
        [(Dense, [20], {'activation': 'relu', 'input_dim': input_nodes.shape[1]}),
         (Dense, [20], {'activation': 'relu', 'input_dim': 20}),
         (Dense, [20], {'activation': 'relu', 'input_dim': 20}),
         (Dense, [20], {'activation': 'relu', 'input_dim': 20}),
         (Dense, [outputN], {'activation': 'softmax', 'input_dim': 20})
        ],
        # Add more models here...  [(...), (...), ...], [(...), ...], ....
        # 3 layer wide 50x50xoutput model - With L1 regularization
        [(Dense, [50],{'activation': 'relu', 'input_dim':input_nodes.shape[1],'kernel_regularizer': regularizers.l1(0.01)}),
         (Dense, [50],{'activation': 'relu', 'kernel_regularizer': regularizers.l1(0.01)}),
         (Dense, [outputN], {'activation': 'softmax'})
         ],
        # 6 layer deep net model - With L1 regularization and dropout before the 1st hidden layer
        [(Dropout, [0.2], {'input_shape': (input_nodes.shape[1],), 'noise_shape': None}),
         (Dense, [20], {'activation': 'relu', 'kernel_regularizer': regularizers.l1(0.01)}),
         (Dense, [20], {'activation': 'relu', 'kernel_regularizer': regularizers.l1(0.01)}),
         (Dense, [20], {'activation': 'relu', 'kernel_regularizer': regularizers.l1(0.01)}),
         (Dense, [20], {'activation': 'relu', 'kernel_regularizer': regularizers.l1(0.01)}),
         (Dense, [outputN], {'activation': 'softmax'})
         ],
        # 5 layer deep net model 20x20x20xoutput - With L2 regularization
        [(Dense, [20], {'activation':'relu', 'input_dim':input_nodes.shape[1],'kernel_regularizer': regularizers.l2(0.02)}),
         (Dense, [20], {'activation': 'relu', 'kernel_regularizer': regularizers.l2(0.02)}),
         (Dense, [20], {'activation': 'relu', 'kernel_regularizer': regularizers.l2(0.02)}),
         (Dense, [20], {'activation': 'relu', 'kernel_regularizer': regularizers.l2(0.02)}),
         (Dense, [outputN], {'activation': 'softmax'})
         ],
        # 6 layer deep net model - With L2 regularization and dropout before 1st layer
        [(Dropout, [0.2], {'input_shape': (input_nodes.shape[1],), 'noise_shape': None}),
         (Dense, [30],{'activation': 'relu', 'kernel_regularizer': regularizers.l2(0.02)}),
         (Dense, [30], {'activation': 'relu', 'kernel_regularizer': regularizers.l2(0.02)}),
         (Dense, [30], {'activation': 'relu', 'kernel_regularizer': regularizers.l2(0.02)}),
         (Dense, [30],{'activation': 'relu', 'kernel_regularizer': regularizers.l2(0.02)}),
         (Dense, [outputN], {'activation': 'softmax'})
         ]
      ]

    # Extract featurees, get labels, run cross validated models...
    input_nodes = extract_features_from_corpus(files_train, adv_ms, len_ms, offset_s, pca=pca,pca_axes_N=40)
    models = model_list(input_nodes)

    # do something useful with cross validation, e.g. generate tables/graphs, etc.
    # HR:creating arch. list for the models
    arch_list = []
    for item in models:
        # hr: using cross validator on the extracted feature from the corpus data
        arch_list.append(CrossValidator(input_nodes, get_class(files_train), item, epochs=100))

    # HR: generating a list of mean error rates for all the implemented models based on training data
    mean_error_rate_list = []
    model_count = 1
    for error_data in arch_list:
        mean_error_rate_list.append(np.mean(error_data.get_errors()))
        print("Mean error of Model%d on training data:"%model_count)
        print(np.mean(error_data.get_errors()))
        model_count += 1

    #hr: Now evaluating test data
    print("\nModel evaluation on test data...\n")
    #hr: Extracting test data features
    test_data_input = extract_features_from_corpus(files_test, adv_ms, len_ms, offset_s, pca=pca, pca_axes_N=40)
    # hr: converting the scalar labels to vector
    test_label_vector = np_utils.to_categorical(get_class(files_test))

    # var to hold error rate and accuracy based on evaluation data for given models
    model_ac_dict = {} # dict representing the accuracy of all the layers of the models which are dict keys
    key_count = 1 # key count representing models
    error_list_all_models = [] # list of lists of error rate for each layer of all the models
    mean_acc_all_model = [] # list to hold all the mean accuracy of the models
    mean_err_rate_all_model = [] # list to hold all the error rates of all the models

    # creating a dict of all the models accuracy and a list of all model error rates
    for model in arch_list:
        temp_list = []
        temp_error_list = []
        for model_arch in model.get_models():
            acc_res = model_arch.evaluate(test_data_input,test_label_vector,verbose=0)
            temp_list.append(acc_res[1])
            temp_error_list.append(1-acc_res[1])
        model_ac_dict[key_count] = temp_list
        mean_acc_all_model.append(np.mean(temp_list))
        error_list_all_models.append(temp_error_list)
        mean_err_rate_all_model.append(np.mean(temp_error_list))
        # printing error rates for all the models
        print("Mean error for Model%d:" % key_count, np.mean(temp_error_list))
        print("Mean accuracy for model%d:" % key_count, np.mean(temp_list))
        key_count += 1

    # plotting the models v/s mean error rate to show the best model for given data set based given data
    plt.figure()
    plt.plot(mean_err_rate_all_model)
    plt.plot(key_count)
    plt.xlim(0,5)
    plt.ylim(0.0,1.0)
    plt.title("Mean error rate plot for all models")
    plt.ylabel("error_rate")
    plt.xlabel("model #")
    plt.legend(["Mean Error rate of Test data"],loc="upper right")
    plt.show()

    print("All models accuracy dict:",model_ac_dict)
    print("All models error rate:",error_list_all_models)

    # based on the results from the above implemented models; the model 6 produces is best suited for given data set
    # printing all the results for the given model 6

    print("\n---------------------\n")
    print("Printing results of best model 6...\n")
    fold_count = 0
    error_rate_eval = []
    for item in arch_list[5].get_models():
        print(item.summary()) # printing summary of each node for model 6
        result = item.evaluate(test_data_input,test_label_vector,verbose=0)
        error_rate_eval.append(1-result[1])
        print("Fold:",fold_count)
        print("accuracy on test data:",result[1])
        print("Error rate:",1-result[1])
        fold_count += 1
    print("Mean error rate of the model 6:",np.mean(error_rate_eval))
    print("\n---------------------\n")


if __name__ == '__main__':
    # plt.ion()
    main()
