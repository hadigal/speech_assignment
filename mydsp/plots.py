# global python imports
import numpy as np
import os,sys
from scipy.io import wavfile as wf
from scipy import signal as sg
import matplotlib.pyplot as plt
from mydsp import utils as ut

# local imports
from mydsp import audioframes,dftstream,multifileaudioframes

def spectrogram(files, adv_ms, len_ms, plot=False):
    """spectrogram(files, adv_ms, len_ms, plot)
    Given a list of files and framing parameters (advance, length in ms), 
    compute a spectrogram that spans the files.  
    
    Returns [intensity_dB, taxis, faxis]
    
    If optional plot is True (default False), the spectrogram is plotted
    """
    #return 1

    # file = "C:\Hrishi_documents\SDSU\FALL_2018\CS_682\HW2\shaken.wav"  # file path for the wavfile

    # audioFrame = audioframes.AudioFrames(files, adv_ms,len_ms)

    #  creating class instance for MultiFileAudioFrames class

    audioFrame = multifileaudioframes.MultiFileAudioFrames(files,adv_ms,len_ms)

    # getting dft stream instance from dftstream module
    dft_str_inst = dftstream.DFTStream(audioFrame)

    itr = 0  # creating itr var
    time_ax = []  # list to save the time axis values
    start_time = 0  # time advance seeking

    # extracting dft stream from the class instance
    dft_mag_db = []
    for dft in dft_str_inst:
        dft_mag_db.append(dft)

    # generating time axis values
    while itr < len(dft_mag_db):
        time_ax.append(start_time)
        start_time += (adv_ms / 1000)
        itr += 1

    t_ax = np.array(time_ax)
    # getting frequency axis values
    frq_ax = dft_str_inst.get_Hz()
    return [dft_mag_db,t_ax,frq_ax]