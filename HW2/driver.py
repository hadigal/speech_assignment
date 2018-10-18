"""
Name: driver.py [assignment 2]
Author: Hrishikesh Adigal
Red ID: 819708988
Desc.: Write any helper functions you need here
"""

# Global python imports
import matplotlib.pyplot as plt
import numpy as np
# local python imports
from mydsp import plots as myplt
from mydsp import utils as ut
from mydsp import multifileaudioframes,dftstream,audioframes,rmsstream
from mydsp import pca
import os
import sklearn.mixture

def plot_narrowband_wideband(filename):
    """plot_narrowband_widband(filename)
    Given a speech file, display narrowband and wideband
    spectrograms of the speech.
    """
    # wide band spectrogram values
    wide_mag, wide_t_axis, wide_f_axis = myplt.spectrogram\
        ([filename],
         10, 20)
    wide_mag_trans = np.transpose(wide_mag)

    # narrow band spectrogram values
    narrow_mag, narrow_t_axis, narrow_f_axis = myplt.spectrogram \
        ([filename],
         20, 40)
    narrow_mag_trans = np.transpose(narrow_mag)

    plt.figure()
    # wide band plot
    plt.subplot(211)
    temp = plt.pcolormesh(wide_t_axis, wide_f_axis, wide_mag_trans)
    plt.title("Wide band Spectrogram")
    plt.xlabel('time in sec')
    plt.ylabel('freq. in Hz')
    clr_bar_temp = plt.colorbar(temp)
    clr_bar_temp.set_label("Intensity dB")
    # narrow band plot
    plt.subplot(212)
    temp = plt.pcolormesh(narrow_t_axis, narrow_f_axis, narrow_mag_trans)
    plt.title("Narrow band Spectrogram")
    plt.xlabel('time in sec')
    plt.ylabel('freq. in Hz')
    clr_bar_temp = plt.colorbar(temp)
    clr_bar_temp.set_label("Intensity dB")
    plt.tight_layout()
    plt.show()

    
def pca_analysis(corpus_dir):
    """pca_analysis(corpus_dir)
    Given a directory containing a speech corpus, compute spectral features
    and conduct a PCA analysis of the data.
    """
    # computing spectrogram data for multifiles data
    multi_frm_inst = multifileaudioframes.MultiFileAudioFrames(corpus_dir,10,20)
    dft_str_inst = dftstream.DFTStream(multi_frm_inst)

    # list of dft stream
    dft_list = []
    for dft_data in dft_str_inst:
        dft_list.append(dft_data)

    dft_list = np.array(dft_list)
    pca_data = pca.PCA(dft_list)

    # from lect. notes : first m dimensions contain the variance
    # represented by the sum of their eigen
    # values. Getting var captured across the components
    var_diff = (np.cumsum(pca_data.eig_val) / np.sum(pca_data.eig_val))
    # plotting the pca component v/s amt. of var captured for each component
    plt.figure()
    plt.plot(var_diff)
    plt.xlabel('Components')
    plt.ylabel('Variance for each comp.')
    plt.show()

    # sorting the var_diff to create a list of variance
    var_diff = sorted(var_diff)
    # Now computing the number of components needed for each decile of variance
    count = 0
    num_com = []
    var_count = 0
    # iterating the over variance to find data components present in that range
    for var in var_diff:
        while var >= 0.1*(var_count+1):
            if 0.1 * (var_count + 1) > 0.9:
                break
            num_com.append(count+1)
            var_count += 1
        count += 1

    print("Printing the # of components need for each decile[10%-90%] of variance :",num_com)

    # creating the file for 6a.wav file
    file_path = os.path.join("C:\\",woman_dir)
    # creating new instances of multiframe and dftream class
    multi_frm_inst2 = multifileaudioframes.MultiFileAudioFrames([file_path],10,20)
    dft_str_inst2 = dftstream.DFTStream(multi_frm_inst2)

    # list of dft stream
    dft_list2 = []
    for dft_data2 in dft_str_inst2:
        dft_list2.append(dft_data2)
    dft_list2 = np.array(dft_list2)

    # get reduced dim PCA transform data; using only enough components to capture 60% of the data.
    transform_data = pca_data.transform(dft_list2,num_com[5])

    # computing the time axis with dft stream data
    t_ax = 10 * len(dft_list2)
    t_axis = []
    for itr in range(0, t_ax, 10):
        t_axis.append(itr/1000)
    t_axis = np.array(t_axis)

    # computing the PCA transform on the 6a.wav file
    pca_trans = np.transpose(transform_data)
    # converting the vector values to abs. value
    pca_trans_mag = np.abs(pca_trans)

    # computing the y axis; plotting the spectrogram with PCA comp as y axis
    y_axis = []
    for itr in range(0,len(pca_trans_mag)):
        y_axis.append(itr)
    y_axis = np.array(y_axis)

    # plotting the spectrogram for t vs pca component
    plt.figure()
    temp = plt.pcolormesh(t_axis, y_axis, pca_trans_mag)
    plt.title("Spectrogram for time v/s PCA")
    plt.xlabel('time in sec')
    plt.ylabel('PCA Components')
    clr_bar_temp = plt.colorbar(temp)
    clr_bar_temp.set_label("Intensity dB")
    plt.show()

def speech_silence(filename):
    """speech_silence(filename)
    Given speech file filename, train a 2 mixture GMM on the
    RMS intensity and label each frame as speech or silence.
    Provides a plot of results.
    """
    # creating instance of audioframe and rmsstream
    audio_frm_inst = audioframes.AudioFrames(filename,10,20)
    rms_inst = rmsstream.RMSStream(audio_frm_inst)

    # getting rms intensity stream from the audio frame data
    rms_strm = [rms_val for rms_val in rms_inst]
    rms_strm = np.array(rms_strm)

    # taking transpose
    reshape_rms = np.reshape(rms_strm, (-1,1))
    # creating a gaussian mixture model
    gmm = sklearn.mixture.GaussianMixture(n_components=2)
    gmm.fit(reshape_rms)
    # predict using gaussian model
    predit = gmm.predict(reshape_rms)

    #creating time axis
    st_time = 0
    itr = 0
    t_ax = []
    while itr < len(rms_inst):
        t_ax.append(st_time)
        st_time += audio_frm_inst.get_frameadv_ms()/1000
        itr += 1
    t_ax = np.array(t_ax)

    # separating speech and noise
    speech = []
    sp_t_ax = []
    noise = []
    n_t_ax = []
    m_x,m_y = gmm.means_
    sp_flg = None
    # setting flages for speech and noise based on the gmm mean values
    # higher the mean higher the content
    if m_x > m_y:
        sp_flg = 0
    else:
        sp_flg = 1

    for itr2 in range(predit.size):
        # separating based on the flags set
        if predit[itr2] == sp_flg:
            speech.append(reshape_rms[itr2])
            sp_t_ax.append(t_ax[itr2])
        else:
            noise.append(reshape_rms[itr2])
            n_t_ax.append(t_ax[itr2])
    speech = np.array(speech)
    noise = np.array(noise)

    # plotting the speech vs noise content graph
    plt.figure()
    plt.scatter(sp_t_ax, speech, marker='x', label='Speech content')
    plt.scatter(n_t_ax, noise, marker='o', label='Noise content')
    plt.title("rms intensity w.r.t. time for speech-noise distribution")
    plt.xlabel("time in sec")
    plt.ylabel('rms int. dB')
    plt.legend()
    plt.show()

    
if __name__ == '__main__':
    # If we are here, we are in the script-level environment; that is
    # the user has invoked python driver.py.  The module name of the top
    # level script is always __main__
    # plt.ion()

    # problem 2
    # call plot_narrowband_wideband
    file_shaken = 'C:\Hrishi_documents\SDSU\FALL_2018\CS_682\HW2\solution\shaken.wav'
    plot_narrowband_wideband(file_shaken)

    # problem 3
    # Call pca_analysis
    files  = ut.get_corpus('C:\Hrishi_documents\SDSU\FALL_2018\CS_682\HW2\woman')
    woman_dir = "Hrishi_documents\SDSU\FALL_2018\CS_682\HW2\woman\\ac\\6a.wav"
    pca_analysis(files)

    # problem 4
    # call speech_silence
    speech_silence(file_shaken)