'''
utils - Miscellaneous utility functions
'''


import os
import os.path

        
def get_corpus(dir, filetype=".wav"):
    """get_corpus(dir, filetype=".wav"
    Traverse a directory's subtree picking up all files of correct type
    
    A speech corpus is a collection (corpus is Latin for body) of speech data 
    This returns a list of files in a corpus when the corpus is contained within
    a directory tree.
    """
    
    files = []
    
    # Standard traversal with os.walk, see library docs
    for dirpath, dirnames, filenames in os.walk(dir):
        for filename in [f for f in filenames if f.endswith(filetype)]:
            files.append(os.path.join(dirpath, filename))
                         
    return files
    