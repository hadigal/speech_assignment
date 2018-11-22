'''
@author: mroch
'''

from .pca import PCA
from .multifileaudioframes import MultiFileAudioFrames
from .dftstream import DFTStream
from .rmsstream import RMSStream
from .audioframes import AudioFrames
 

# Standard Python libraries
import os.path
from datetime import datetime
# Add-on libraries
import numpy as np
import matplotlib.pyplot as plt


import hashlib  # hash functions
from librosa.feature.spectral import melspectrogram
from statsmodels.tsa.x13 import Spec
from .endpointer import Endpointer

def s_to_frame(s, adv_ms):
    """s_to_frame(s, adv_ms) 
    Convert s in seconds to a frame index assuming a frame advance of adv_ms
    """
    
    return np.int(np.round(s * 1000.0 / adv_ms))

def plot_matrix(matrix, xaxis=None, yaxis=None, xunits='time (s)', yunits='Hz', zunits='(dB rel.)'):
    """plot_matrix(matrix, xaxis, yaxis, xunits, yunits
    Plot a matrix.  Label columns and rows with xaxis and yaxis respectively
    Intensity map is labeled zunits.
    Put "" in any units field to prevent plot of axis label
    
    Default values are for an uncalibrated spectrogram and are inappropriate
    if the x and y axes are not provided
    """
    
    if xaxis is None:
        xaxis = [c for c in range(matrix.shape[1])]
    if yaxis is None:
        yaxis = [r for r in range(matrix.shape[0])]
        
    # Plot the matrix as a mesh, label axes and add a scale bar for
    # matrix values
    plt.pcolormesh(xaxis, yaxis, matrix)
    plt.xlabel(xunits)
    plt.ylabel(yunits)
    plt.colorbar(label=zunits)
    
def spectrogram(files, adv_ms, len_ms, specfmt="dB", mel_filters_N=12):
    """spectrogram(files, adv_ms, len_ms, specfmt)
    Given a filename/list of files and framing parameters (advance, length in ms), 
    compute a spectrogram that spans the files.
    
    Type of spectrogram (specfmt) returned depends on DFTStream, see class
    for valid arguments and interpretation, defaults to returning
    intensity in dB.
    
    Returns [intensity, taxis_s, faxis_Hz]
    """

    # If not a list, make it so number one...
    if not isinstance(files, list):
        files = [files]
        
    # Set up frame stream and pass to DFT streamer
    framestream = MultiFileAudioFrames(files, adv_ms, len_ms)
    dftstream = DFTStream(framestream, specfmt=specfmt, mels_N = mel_filters_N)
    
    # Grab the spectra
    spectra = []
    for s in dftstream:
        spectra.append(s)
        
    # Convert to matrix
    spectra = np.asarray(spectra)
        
    # Time axis in s
    adv_s = framestream.get_frameadv_ms() / 1000    
    t = [s * adv_s for s in range(spectra.shape[0]) ]
    
    return [spectra, t, dftstream.get_Hz()]


def fixed_len_spectrogram(file, adv_ms, len_ms, offset_s, specfmt="dB", 
                          mel_filters_N = 12):
    """fixed_len_spectrogram(file, adv_ms, len_ms, offset_s, specfmt, 
        mel_filters_N)
        
    Generate a spectrogram from the given file.
    Truncate the spectrogram to the specified number of seconds
    
    adv_ms, len_ms - Advance and length of frames in ms
    
    offset_s - The spectrogram will be truncated to a fixed duration,
        centered on the median time of the speech distribution.  The
        amount of time to either side is determned by a duration in seconds,
        offset_s.  
        
        The speech is endpointed using an RMS energy endpointer
        and centered median time of frames marked as speech.  If the fixed
        duration is longer than the available speech, random noise frames
        are drawn from sections marked as noise to complete the spectrogram
        
    specfmt - Spectrogram format. See dsp.dftstream.DFTStream for valid formats
    
    mel_filters_N - Number of Mel filters to use when specft == "Mel"
    """
    
    # TODO:
    # Use the Endpointer class to determine the times associated with speech.

    # Find the median of the frames marked as speech

    # Generate a spectrogram of the appropriate type (and number of Mel filters 
    # if needed).

    # Pull out median -/+ offset_s

    # Pad the left and right sides with zeros if too short.
    # Return the spectrogram along with time and frequency axes
    # The time axis should reflect the original times, e.g. if offset_s is .25 and
    # the center frame is at .5 s, the time axis should run from .25 to .75 s

    # hr: creating endptr obj
    endptr_obj = Endpointer(file,adv_ms,len_ms)
    # hr: getting the speech frames indexs
    speech_frms = endptr_obj.speech_frames()
    speech_idx = []
    for idx in range(0,len(speech_frms) -1):
        if speech_frms[idx] == True:
            speech_idx.append(idx)
    # spectra for entire file
    spectra,t_s,f_hz = spectrogram(file,adv_ms,len_ms,specfmt,mel_filters_N)
    # HR: np array for padding the truncated values if necessary
    pad_arr = np.zeros([len(spectra[0])])
    # HR: taking median of the speech frame index obtained from endpointer
    median_sp_frms = int(np.median(speech_idx))
    # HR: padding logic for start and end of the axis
    low_lim = t_s[median_sp_frms] - offset_s
    upper_lim = t_s[median_sp_frms]+offset_s
    # arrays to save the truncated values of time axis and spectra
    trunc_time = []
    trunc_spec = []
    # final iterable index
    max_idx = len(t_s) - 1
    # padding logic...
    # padding zeros on the left side
    if low_lim < t_s[0]:
        r_time = low_lim/(adv_ms/1000)
        for itr in range(round(r_time)+1,0):
            trunc_time.append(itr*(adv_ms/1000))
            trunc_spec.append(pad_arr)
    # padding zeros on the right side
    elif upper_lim > t_s[max_idx]:
        r_time = upper_lim/(adv_ms/1000)
        up_time = t_s[max_idx]/(adv_ms/1000)
        for itr in range(round(up_time)+1,round(r_time)+1):
            trunc_time.append(itr*(adv_ms/1000))
            trunc_spec.append(pad_arr)
    itr = 0
    # extracting truncated spectra from the entire spectra obtained
    while itr < len(spectra):
        if ((round(t_s[itr],2) > round(low_lim,2)) and (round(t_s[itr],2) < round(upper_lim,2))):
            trunc_spec.append(spectra[itr])
            trunc_time.append(t_s[itr])
        itr += 1
    trunc_time = np.array(trunc_time)
    trunc_spec = np.array(trunc_spec)

    return [trunc_spec, trunc_time, f_hz]
    
def pca_analysis_of_spectra(files, adv_ms, len_ms, offset_s): 
    """"pca_analysis_of_spectra(files, advs_ms, len_ms, offset_s)
    Conduct PCA analysis on spectra of the given files
    using the given framing parameters.  Only retain
    central -/+ offset_s of spectra
    """
    md5 = hashlib.md5()
    string = "".join(files)
    md5.update(string.encode('utf-8'))
    hashkey = md5.hexdigest()
    
    filename = "VarCovar-" + hashkey + ".pcl"
    try:
        pca = PCA.load(filename)

    except FileNotFoundError:
        example_list = []
        for f in files:
            [example, _t, _f] = fixed_len_spectrogram(f, adv_ms, len_ms, offset_s, "dB")
            example_list.append(example)
            
        # row oriented examples
        spectra = np.vstack(example_list)
    
        # principal components analysis
        pca = PCA(spectra)

        # Save it for next time
        pca.save(filename)
        
    return pca


       
def get_corpus(dir, filetype=".wav"):
    """get_corpus(dir, filetype=".wav"
    Traverse a directory's subtree picking up all files of correct type
    """
    
    files = []
    
    # Standard traversal with os.walk, see library docs
    for dirpath, dirnames, filenames in os.walk(dir):
        for filename in [f for f in filenames if f.endswith(filetype)]:
            files.append(os.path.join(dirpath, filename))
                         
    return files
    
def get_class(files):
    """get_class(files)
    Given a list of files, extract numeric class labels from the filenames
    """
    
    # TIDIGITS single digit file specific
    
    classmap = {'z': 0, '1': 1, '2': 2, '3': 3, '4': 4,
                '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'o': 10}

    # Class name is given by first character of filename    
    classes = []
    for f in files:        
        dir, fname = os.path.split(f) # Access filename without path
        classes.append(classmap[fname[0]])
        
    return classes
    
class Timer:
    """Class for timing execution
    Usage:
        t = Timer()
        ... do stuff ...
        print(t.elapsed())  # Time elapsed since timer started        
    """
    def __init__(self):
        "timer() - start timing elapsed wall clock time"
        self.start = datetime.now()
        
    def reset(self):
        "reset() - reset clock"
        self.start = datetime.now()
        
    def elapsed(self):
        "elapsed() - return time elapsed since start or last reset"
        return datetime.now() - self.start
    
