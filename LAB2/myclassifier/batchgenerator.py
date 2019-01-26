'''
Created on Nov 26, 2017

@author: mroch
'''
import numpy as np
import math
from keras.utils import to_categorical

class PaddedBatchGenerator:
    """PaddedBatchGenerator
    Class for sequence length normalization.
    Each sequence is normalized to the longest sequence in the batch
    or a specified length by zero padding. 
    """
    
    debug = False
    
    @classmethod
    def longest_sequence(cls, features):
        """longest_sequence(features)
        Return the longest sequence in a set of features
        
        # Arguments
        features:  list of arrays, each one is a time x dim array
        """
        # HR: Creating a list to store the length of all the features
        f_list = []
        for feature in features:
            f_list.append(feature.shape[0])
        # HR: using max function to find the biggest element of the list
        max_len = max(f_list)
        return max_len
    
    def __init__(self, corpus, utterances, batch_size=100):
        """"PaddedBatchGenerator(corpus, utterances, batch_size)
        Create object for generating padded batches.
        
        # Arguments
        corpus - A corpus object such as an instance of timit.Corpus
        utterances - A list of utterances to iterate over.
        batch_size - Number of examples in the batch 
        """
        self.corpus = corpus
        self.utterances = utterances

        # HR: Creating a lists to hold features, labels
        self.features = []
        self.labels = []
        # HR: temp directory to hold phoneme
        self.ph_vect_labels = []
        # HR: count variable to hold the current utterance count
        count = 0
        # HR: Now, iter over the list of utterance to extract phone for every set frame
        # this is done by iterating over the list of features and labels and extracting
        # phones for every frame with help of stop time for every frame returned with
        # get_label() method. Before appending the phoneme label matrix with phonemes
        # for each frame converted them to 1 hot vector using the phones_to_onehot()
        for i in self.utterances:
            # list of features and labels
            self.features.append(corpus.get_features(i))
            self.labels.append(corpus.get_labels(i))
            # converting the phones returned as list to one hot vector
            ph = corpus.phones_to_onehot(self.labels[count][2])

            # itr over the list of stop time for each frame in a feature and saving the
            # corressponding phone label
            itr = 0
            temp_ph_list = []
            for j in range(len(self.features[count])):
                # print("j:{}".format(j))
                if j > self.labels[count][1][itr]:
                    itr += 1
                # HR: this a debug to catch the feature with stop time less than
                # actual end time of the feature file. If found so then poping the unwanted frames
                if (j == int(self.labels[count][1][itr])) and (itr == (len(ph) - 1)):
                    print("Corrupt Utterance:{}".format(i))
                    # print("GOTCHA:")
                    lim = int(j)
                    self.features[count] = self.features[count][:lim]
                    break
                temp_ph_list.append(ph[itr])

            temp_ph_list = np.array(temp_ph_list)
            self.ph_vect_labels.append(temp_ph_list)
            count += 1

        self.len_features = len(self.features)
        # self.len_labels = len(self.labels)
        self.len_labels = len(self.ph_vect_labels)

        self.wrap_flg = 0
        self.batch_size = batch_size
        self.batch_itr = 0
        self._epoch = 0
        self.b_count = 0
        self.b_count_list = []
        
    @property
    def epoch(self):
        return self._epoch
    
            
    def __iter__(self):
        return self
        
    def __next__(self):
        """__next__() - Return next set of training examples and target values
        All training vectors in batch will be normalized to the longest
        sequence in the batch.
        """

        # HR: Now padding all the features and labels in the batch with the zeros upto the
        # the length of the maximum len feature
        ex = []
        lab = []

        batch_stop = self.batch_itr + self.batch_size
        # checking if stop idx is greater than or equal feature length
        if batch_stop >= self.len_features:
            # Wrapping the features and labels to the len for features
            turn_around = batch_stop % self.len_features
            ex.extend(self.features[self.batch_itr:self.len_features])
            ex.extend(self.features[0:turn_around])
            lab.extend(self.ph_vect_labels[self.batch_itr:self.len_features])
            lab.extend(self.ph_vect_labels[0:turn_around])

            # now pointing the batch_itr to the start of next batch
            self.batch_itr = turn_around
            # incrementing epoch as we start from the beginning again
            self._epoch += 1
            # appending the list to hold batch count per epoch
            self.b_count_list.append(self.b_count)
            # reinit the batch_count var to 0 at the end of epoch
            self.b_count = 0
        else:
            ex.extend(self.features[self.batch_itr:batch_stop])
            lab.extend(self.ph_vect_labels[self.batch_itr:batch_stop])
            # incrementing batch_itr to the beginning of next batch
            self.batch_itr = batch_stop
            self.b_count += 1

        # ########
        # if self.batch_itr == self.len_features:
        #     self.b_count_list.append(self.b_count)
        #     self._epoch = self._epoch + 1
        #     self.b_count = 0
        #     self.batch_itr = 0
        #
        # batch_stop = self.batch_itr + self.batch_size
        # fwd = batch_stop
        # # checking
        # if batch_stop > self.len_features:
        #     #
        #     turn_around = batch_stop%self.len_features
        #     # ex.append(self.features[0:turn_around])
        #     # lab.append(self.labels[0:turn_around])
        #     ex.extend(self.features[0:turn_around])
        #     # lab.extend(self.ph_labels[0:turn_around])
        #     # lab.extend(self.temp_ph_labels[0:turn_around])
        #     lab.extend(self.ph_vect_labels[0:turn_around])
        #
        #
        #     # self.batch_itr = turn_around
        #     fwd = turn_around
        #     # self.wrap_flg = 1
        #     batch_stop = self.len_features
        #     self.b_count_list.append(self.b_count)
        #     self._epoch = self._epoch +1
        #     self.b_count = 0
        #
        # # ex.append(self.features[self.batch_itr:batch_stop])
        # # lab.append(self.labels[self.batch_itr:batch_stop])
        # ex.extend(self.features[self.batch_itr:batch_stop])
        # # lab.extend(self.ph_labels[self.batch_itr:batch_stop])
        # # lab.extend(self.temp_ph_labels[self.batch_itr:batch_stop])
        # lab.extend(self.ph_vect_labels[self.batch_itr:batch_stop])
        #
        # self.batch_itr = fwd
        # # if self.wrap_flg != 1:
        # #     self.batch_itr = batch_stop
        # # else:
        # #     self.wrap_flg = 0
        # self.b_count += 1

        # else:
        #     ex.append(self.features[self.batch_itr:batch_stop])
        #     lab.append(self.labels[self.batch_itr:batch_stop])
        #     self.batch_itr = batch_stop
        #     self.b_count += 1

        # now that we have features and labels for each batch; implementing a logic to
        # pad zeros to the feature row of feature matrix which is less than len of
        # the len of longest feature row
        max_seq_len = self.longest_sequence(ex)
        batch_examples = np.zeros([self.batch_size,max_seq_len,ex[0].shape[1]])

        # getting silence frame labels to append the batch labels to the length of
        # batch feature length
        sLab = self.corpus.get_silence()
        sLabVec = self.corpus.phones_to_onehot([sLab])
        vecLen = sLabVec[0]
        # Padding zeros to the label matrix for silence frame labels
        batch_labels = np.zeros([self.batch_size, max_seq_len, len(vecLen)])

        # assigning the silence frame batch label vectors
        # itr1 = 0
        # while itr1 < len(batch_labels):
        for itr1 in range(len(batch_labels)):
            # itr2 = 0
            # while itr2 < len(batch_labels[itr1]):
            for itr2 in range(len(batch_labels[itr1])):
                batch_labels[itr1][itr2] = vecLen
            #     itr2 += 1
            # itr1 += 1

        count = 0
        # now that padding is done assigning values at remaining indices of the batch
        # feature and label matrices
        for itr in range(self.batch_size):
            temp = ex[itr].shape[0]
            batch_examples[itr, -temp:, :] = ex[itr]
            batch_labels[itr,-temp:,:] = lab[itr]
            count += 1

        return (batch_examples, batch_labels)
    
    def get_batches_per_epoch(self):
        "get_batches_per_epoch() - Approx number of batches per epoch"
        # As mentioned; taking the ceil of feature len divided by batch size
        len_var = len(self.features)
        batch_per_epoch = int(math.ceil(len_var/self.batch_size))
        return batch_per_epoch