"""
Name: rmsstream.py
Author: Hrishikesh Adigal
Red ID: 819708988
"""

# Add imports as needed
import numpy as np

class RMSStream:
    """RMSSTream
    Streamer object for root mean square intensity in dB
    Given an object of class AudioFrames
    """
    
    def __init__(self, frames):
        "RMSSTream - root mean square stream from an AudioFrames object"
        self.count = 0 #iterator var
        self.all_frames = frames # getting frames from the Audioframes
        self.total_frame = len(self.all_frames)
        self.rms = []

        #computing the rms value for all the frames
        for frame in self.all_frames:
            self.rms.append(10*np.log10(np.mean(np.square(frame/1.0))))
        
    def __iter__(self):
        "__iter_ - Return an object that iterates over RMS frame values"
        
        # Implementation decision
        # You can return self and implement a __next__(self) in this class
        # or you can create and return an instance of an iteration class
        # of your design that supports __next__(self).  In either case, the
        # __next__ method will return the next RMS value in dB rel.
        return self

    def __next__(self):
        """
        Implemented the __next__() for the iterator object
        """
        if self.count >= self.total_frame:
            raise StopIteration
        else:
            self.count += 1
            return self.rms[self.count -1]

    def shape(self):
        "shape() - shape of tensor generated by iterator"
        
        # See descriptions in audioframes.py
        return [1,1]
    
    def size(self):
        "size() - number of elements in tensor generated by iterator"
        return 1
    
    def __len__(self):
        "__len__() - Number of frames in stream"
