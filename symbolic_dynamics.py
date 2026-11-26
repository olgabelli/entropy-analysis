
# Symbolic dynamics

# Functions:
# 1. digitization_binary()   - Convert a time series into binary values.
# 2. block_prob_no_overlap() - Estimate block probabilities using lumping (no-overlap).
# 3. block_prob_overlap()    - Estimate block probabilities using overlapping windows.


import numpy as np
from itertools import product

from entropy import prob_in_each_bin


def digitization_binary(x, threshold):
    '''
    Convert a 1D time series into binary values (0,1).

    Parameters
    ----------
    x : Array
        1D time series.
    threshold : float
        Threshold for digitization.

    Returns
    -------
    values: ndarray
        Values >threshold become 1, while values <=threshold become 0.
        NaN values are preserved.

    Raises
    ------
    ValueError
        If threshold is not finite or x is not 1D.
    '''
    
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")
    
    x = np.array(x)
    
    if x.ndim != 1:
        raise ValueError("Input must be 1D array.")
     
    mask = x > threshold
    values = np.where(np.isnan(x), np.nan, mask.astype(int))

    
    return values



def block_prob_no_overlap(digitized_values, 
                          block_length):
    '''
    Estimate probabilities of binary blocks using the lumping method (non-overlapping).
    Incomplete blocks at the end of the sequence are ignored
    and blocks containing NaN are discarded.
    
    Parameters
    ----------
    digitized_values : Array
        1D sequence of 0,1 or NaN.
    block_length : int
        Num. of symbols per block.

    Returns
    -------
    prob : ndarray
        Probability of each possible binary block.
    bins : ndarray
        Bin edges for the decimal block indices.
    actual_block_bins : list of tuples
        Binary patterns corresponding to the probabilities.

    Raises
    ------
    ValueError
        If digitized_values is not 1D or contains values other than 0,1,NaN.
        If block_length is not a positive integer.
    '''
    
    
    digitized_values = np.array(digitized_values)
    
    if digitized_values.ndim != 1:
        raise ValueError("Input must be 1D array.")
    
    mask = (digitized_values != 0) & (digitized_values != 1) & ~np.isnan(digitized_values)
    if np.any(mask):
        raise ValueError("Input must contain only 0, 1, or NaN.")
    
    m = block_length
    if m % 1 != 0 or m<=0:
        raise ValueError("block_length must be positive integer.")
    m = int(m)
    
    not_included = len(digitized_values) % m
    num_of_blocks = len(digitized_values) // m
    

    D = 2 ** np.arange(m-1, -1, -1) # m=3: 2**2, 2**1, 2**0
    
    t = []
    for i in range(0, len(digitized_values)-not_included, m):
        
        block = digitized_values[i:i+m] 
        
        # discard blocks containing NaN values
        if np.any(np.isnan(block)):
            continue
        
        t.append( block @ D) # block[0]*2**2 + block[1]*2**1 + block[2]* 2**0

    t = np.array(t)     
    bins = np.arange(0, 2**m+1, 1) # +1 to include the right edge of the last bin.

    actual_block_bins = list(product([0, 1], repeat=m))
    
    prob = prob_in_each_bin(t, bins)
    
    return prob, bins, actual_block_bins




def block_prob_overlap(digitized_values, 
                       block_length, 
                       n_overlap):
    '''
    Estimate probabilities of binary blocks using overlapping windows.
    Blocks advance by block_length - n_overlap positions.
    Incomplete blocks at the end of the sequence are ignored
    and blocks containing NaN are discarded.
    
    Parameters
    ----------
    digitized_values : Array
        1D sequence of 0,1 or NaN.
    block_length : int
        Num. of symbols per block.
    n_overlap : int
        Number of symbols shared by consecutive blocks.
        Must be between 0 and block_length - 1.


    Returns
    -------
    prob : ndarray
        Probability of each possible binary block.
    bins : ndarray
        Bin edges for the decimal block indices.
    actual_block_bins : list of tuples
        Binary patterns corresponding to the probabilities.

    Raises
    ------
    ValueError
        If digitized_values is not 1D or contains values other than 0,1,NaN.
        If block_length is not a positive integer.
        If n_overlap is not an integer between 0 and block_length-1.

    '''
    
    digitized_values = np.array(digitized_values)
    if digitized_values.ndim != 1:
        raise ValueError("Input must be 1D array.")
    
    mask = (digitized_values != 0) & (digitized_values != 1) & ~np.isnan(digitized_values)
    if np.any(mask):
        raise ValueError("Input must contain only 0, 1, or NaN.")
    
    
    m = block_length
    if m % 1 != 0 or m<=0:
        raise ValueError("block_length must be positive integer.")
    m = int(m)
    
    
    if not np.isfinite(n_overlap) or n_overlap % 1 != 0:
        raise ValueError("n_overlap must be an integer.")
    n_overlap = int(n_overlap)
    
    if n_overlap < 0 or n_overlap >= m:
        raise ValueError("n_overlap must be between 0 and block_length-1.")
        
    D = 2 ** np.arange(m-1, -1, -1) # m=3: 2**2, 2**1, 2**0
    
    t = []
    for i in range(0, len(digitized_values)-m+1, m-n_overlap):
        
        block = digitized_values[i:i+m] 
        
        # discard blocks containing NaN values
        if np.any(np.isnan(block)):
            continue

        t.append(block @ D)
        
    t = np.array(t) 
    bins = np.arange(0, 2**m+1, 1) # +1 to include the right edge of the last bin.
    actual_block_bins = list(product([0, 1], repeat=m))
    
    prob = prob_in_each_bin(t, bins)

    return prob, bins, actual_block_bins


  
    
    