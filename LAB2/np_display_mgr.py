'''
np_display_mgr - utility for printing numpy arrays
'''

import numpy as np

class NumpyDisplayMgr:
    """Control how numpy prints things
    Minor modifications from Paul Price's class:
    https://stackoverflow.com/questions/1987694/how-to-print-the-full-numpy-array/24542498#24542498
    
    Useful for printing a whole numpy matrix without the ...
    Example:
        with NumpyDisplayMgr():
            print(some_matrix)    # be careful what you wish for...
        
        
    """

    def __init__(self, **kwargs):
        """NumPyDisplayMgr(keywor args)
        Permits any keyword argument for numpy.set_printoptions
        If threshold not specified sets to Inf
        """
        
        if 'threshold' not in kwargs:
            kwargs['threshold'] = np.Inf
        self.opt = kwargs

    def __enter__(self):
        
        self._previous_opt = np.get_printoptions()  # Save current options
        np.set_printoptions(**self.opt)

    def __exit__(self, type, value, traceback):
        # Reset to previous
        np.set_printoptions(**self._previous_opt)    