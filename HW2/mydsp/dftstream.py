"""
dftstream - Streamer for Fourier transformed spectra
"""
# global python imports
import numpy as np
import scipy.signal as signal

# Hr: Taking objects from AudioFrames class from audioframes.py
     
class DFTStream:
    """
    DFTStream - Transform a frame stream to various forms of spectra
    """

    def __init__(self, frame_stream, specfmt="dB"):
        """
        DFTStream(frame_stream, specfmt)
        Create a stream of discrete Fourier transform (DFT) frames using the
        specified sample frame stream. Only bins up to the Nyquist rate are
        returned in the stream Optional arguments:

        specfmt - DFT output:
            "complex" - return complex DFT results
             "dB" [default] - return power spectrum 20log10(magnitude)
             "mag^2" - magnitude squared spectrum
        """
        # HR: taking instance of framestream class
        self.frm_st_ins = frame_stream
        # HR: totals frames from the frame stream\
        self.big_N = self.frm_st_ins.get_frameadv_samples()

        # sampling frequency(Nyquist rate)
        self.sam_freq = self.frm_st_ins.get_Nyquist()

        # HR: creating an array of frequencies associated with each spectral bin of
        # dft stream --> k*Fs/N, where k is each sample number from [0 to N-1]
        self.freq_hz = np.arange(self.big_N) / self.big_N * self.sam_freq

        self.specfmt = specfmt
        self.win_func = None
        self.frame_itr = iter(frame_stream)

    def shape(self):
        """
        shape() - Return dimensions of tensor yielded by next()"
        Returns shape of one spectral vector
        """

        return np.asarray([len(self.freq_hz),1])
    
    def size(self):
        "size() - number of elements in tensor generated by iterator"
        
        # Returns number of elements in one spectral vector
        return np.asarray(np.product(self.shape()))
   
    def get_Hz(self):
        """get_Hz() - Return list of frequencies associated with each 
        spectral bin.  (List is of the same size as the # of spectral
        bins up to the Nyquist rate, or half the frame lenght)
        """

        return self.freq_hz
            
    def __iter__(self):
        "iter() Return iterator for stream"
        return self
    
    def __next__(self):
        "next() Return next DFT frame"

        frm_data = next(self.frame_itr)
        # window function advancing with frame adv. length
        self.win_func = signal.get_window("hamming",frm_data.shape[0])
        # pt by pt multiplication of the frame and win. function
        win_data = frm_data * self.win_func
        dft = np.fft.fft(win_data)
        # returning dft stream as per specified format
        if self.specfmt is "complex":
            return dft
        elif self.specfmt is "dB":
            dft_mag = np.abs(dft)
            dft_mag = dft_mag[:int(len(dft_mag) / 2)]
            dft_mag_db = 20 * np.log10(dft_mag)
            return dft_mag_db
        elif self.specfmt is "mag^2":
            dft_sq_mag = np.square(np.abs(dft))
            dft_sq_mag_val = 10 * np.log10(dft_sq_mag)
            return dft_sq_mag_val

    def __len__(self):
        "len() - Number of tensors in stream"
        return len(self.frm_st_ins)
