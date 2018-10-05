"""
Name: driver.py
Author: Hrishikesh Adigal
Red ID: 819708988
"""

import matplotlib.pyplot as plt
import numpy as np
from mydsp import audioframes,rmsstream

def driver():
    
    # Construct an RMS stream for the file "shaken.wav"
    # with the specified parameters.  Iterate to create a list
    # of frame intensities.  Plot them with time on the abcissa (x)
    # axis in seconds and RMS intensity on the ordinate (y) axis.
    
    # Be sure to listen to the audio to see if the intensity plot looks
    # correct.
    
    # Here's where you write lots of exciting stuff!
    
    # Might want to set a breakpoint here if your windows are closing on exit
    file = "C:\Hrishi_documents\SDSU\FALL_2018\CS_682\HW1\shaken.wav" #file path for the wavfile
    audioFrame = audioframes.AudioFrames(file,20,10) #creating class instance for AudioFrames class

    rms = rmsstream.RMSStream(audioFrame) #creating class instance of RMSStream class

    itr = 0 #creating itr var
    time_val = [] #list to save the time values
    start_time = 0 #time advance seeking

    while itr < len(rms.rms):
        time_val.append(start_time)
        start_time += (audioFrame.get_frameadv_ms())/1000
        itr += 1

    # plotting RMS energy graph
    plt.figure()
    plt.plot(time_val, rms.rms)
    plt.title("RMS energy plot")
    plt.xlabel("sec")
    plt.ylabel('dB')
    plt.show()

if __name__ == '__main__':
    # If invoked as the main module, e.g. python driver.py, execute
    driver()