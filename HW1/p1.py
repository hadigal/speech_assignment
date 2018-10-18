"""
Name: DFT problem
Author: Hrishikesh Adigal
Red ID: 819708988
"""

import numpy as np
import scipy.io.wavfile as spw
import scipy.signal as sig
from pathlib import Path
import matplotlib.pyplot as plt

# plt.ion() #interactive graphs

# creating a var to hold the wavfile path
filePath = "C:\\Hrishi_documents\\SDSU\\FALL_2018\\CS_682\\HW1\whistle.wav"

#using scipy.io.wavfile to read the wavfile
rate, data = spw.read(filePath)
print ("Sampling rate:%d\n"%rate)
print ("numpy array data:\n")
print (data.T[0])# get the first track from the signal file

sp_rate = rate # sampling rate of the wav file

# computing the DFT on the data from the wavfile

ch1 = data.T[0] # get data only from channel 1
total_sp = int(data.shape[0]) #get the total number of sample data present in the file
print("Total samples:%d"%total_sp)

#normalize data from [-1,1]
#norm_ch1 = [(nData/2**16.)*2-1 for nData in ch1]

#using hamming window to compute the DFT
bins_Hz = np.arange(total_sp)/total_sp*sp_rate
ham_window = sig.get_window("hamming", total_sp)

#multiplying the hamming window function with sample ch data
win_ch = ch1*ham_window
dft = np.fft.fft(win_ch)
# computing the magnitude of the DFT
mag_dft = np.abs(dft)
# converting to dB scale
mag_dft_dB = 20 * np.log10(mag_dft)

# plotting the DFT on freq vs mag scale
plt.figure()
plt.plot(bins_Hz, mag_dft_dB)
plt.title("DFT")
plt.xlabel("Hz")
plt.ylabel('dB')
plt.show()
