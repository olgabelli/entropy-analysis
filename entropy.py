
# Entropy analysis

# Functions:
# 1. logb()                            - Calculate logarithms with a specified base.
# 2. prob_in_each_bin()                - Estimate probabilities using histogram bins.
# 3. check_probabilities()             - Validate a probability distribution.
# 4. entropy_shannon()                 - Calculate Shannon entropy.
# 5. entropy_from_initial_timeseries() - Calculate Shannon entropy from a time series.
# 6. entropy_renyi()                   - Calculate Renyi entropy.
# 7. entropy_tsallis()                 - Calculate Tsallis entropy.


import numpy as np

# %% Helper functions

def logb(x, b):
    '''
    Calculate logarithm of x with base b.

    Parameters
    ----------
    x : Array
        Values for which the logarithm is calculated.
    b : float
        Base of the logarithm. Must be finite, positive, and not equal to 1.

    Returns
    -------
    logb : ndarray 
        log_b(x).

    Raises
    ------
    ValueError
        If b is not finite, positive or is equal to 1.
        If x has any non-positive values.
    '''
    
    x =  np.array(x)
    
    if not np.isfinite(b) or b<=0 or b ==1:
        raise ValueError("The base of the logarithm must be finite, positive and not equal to 1.")
    
    if np.any(x<=0):
        raise ValueError(f"Non-positive values in logb: {np.sum(x<=0)}.")
    
    logb = np.log(x) / np.log(b)
    
    return logb



def prob_in_each_bin(x, bin_edges):
    '''
    Estimate probabilities using histogram bins.

    Parameters
    ----------
    x : Array
        1D array of measurements.
    bin_edges : Array
        Edges of the histogram bins.

    Returns
    -------
    p : ndarray
        Probability of each bin.

    Raises
    ------
    ValueError
        If x is not 1D or is empty.
        If x contains NaN or infinite values.
        If any values fall outside the histogram bins.
    '''
    
    x =  np.array(x)
    
    if x.ndim != 1:
        raise ValueError("Input must be 1D array.")
        
    if len(x) == 0:
        raise ValueError("Input array is empty.")
    
    if not np.all(np.isfinite(x)):
        raise ValueError("Input contains NaN or infinite values.")
        
    counts, bin_edges = np.histogram(x, bins=bin_edges)
    
    if len(x)==np.sum(counts):
        
        p = counts/np.sum(counts)
            
        return p
        
    else:
        raise ValueError("Some values are outside the histogram bins.")
    


def check_probabilities(p):
    '''
    Validate a probability distribution.

    Parameters
    ----------
    p : Array
        Probabilities of possible outcomes.

    Returns
    -------
    p : ndarray
        Validated probabilities.

    Raises
    ------
    ValueError
        If p is empty or not 1D.
        If p has non-finite or negative values.
        If the probabilities do not sum to 1.
    '''

    p = np.asarray(p)

    if p.ndim != 1 or len(p) == 0:
        raise ValueError("Probabilities must be a non-empty 1D array.")

    if not np.all(np.isfinite(p)):
        raise ValueError("Probabilities must be finite.")

    if np.any(p < 0):
        raise ValueError("Probabilities cannot be negative.")

    if not np.isclose(np.sum(p), 1):
        raise ValueError("Probabilities must sum to 1.")

    return p

# %% Shannon Entropy

def entropy_shannon(p, b=np.e, norm = False, n_cases = None):
    '''
    Calculate Shannon entropy.

    Parameters
    ----------
    p : Array
        Probabilities of the possible outcomes.
    b : float, optional
        Base of the logarithm. The default is np.e.
    norm : bool, optional
        If True, normalize entropy (by its maximum value). The default is False.
    n_cases : int, optional
        Total number of possible outcomes. The default is len(p).

    Returns
    -------
    H : float
        Shannon entropy (normalized if norm=True).

    Raises
    ------
    ValueError
        If p is not valid.
        If b is not finite, positive or is equal to 1.
        If n_cases is not an integer >=len(p).
        If normalization is requested with n_cases <= 1.
    '''
    
    p = check_probabilities(p)
    
    if n_cases is None:
        n_cases = len(p)
        
    if n_cases % 1 != 0 or n_cases < len(p):
        raise ValueError("n_cases must be an integer >= len(p).")
    
    if norm and n_cases <= 1:
        raise ValueError("At least two possible outcomes are required for normalization.")

    mask = p>0              # exclude p=0
    p_masked = p[mask]
    
    H = -np.sum(p_masked*logb(p_masked, b))
    
    if norm:
        Hmax = logb(n_cases, b)
        H = H / Hmax
    
    return H



def entropy_from_initial_timeseries(x, bin_edges, b = np.e, norm = False):
    
    '''
    Calculate Shannon entropy directly from a time series using histogram bins.

    Parameters
    ----------
    x : Array
        1D time series.
    bin_edges : Array
        Edges of the histogram bins.
    b : float, optional
        Base of the logarithm. The default is np.e.
    norm : bool, optional
        If True, normalize entropy (by its maximum value). The default is False.
    
    Returns
    -------
    H : float
        Shannon entropy (normalized if norm=True).

    Raises
    ------
    ValueError
        If x is not a valid 1D array or contains NaN or infinite values.
        If any values fall outside the histogram bins.
        If b is not a valid logarithm base.
        If normalization is requested with less than two bins.
    '''
    prob = prob_in_each_bin(x, bin_edges)

    H = entropy_shannon(prob, b, norm, len(bin_edges)-1)    
    
    return H


# %% Renyi entropy

def entropy_renyi(p, a, norm = False, n_cases = None):
    
    '''
    Calculate Renyi entropy.

    Parameters
    ----------
    p : Array
        Probabilities of the possible outcomes.
    a : float
        Free parameter-order a, often 0.5-2. Must be finite and non-negative.
    norm : bool, optional
        If True, normalize entropy (by its maximum value). The default is False.
    n_cases : int, optional
        Total number of possible outcomes. The default is len(p).


    Returns
    -------
    Hr : float
        Renyi entropy (normalized if norm=True).

    Raises
    ------
    ValueError
        If p is not valid.
        If a is negative or not finite.
        If n_cases is not an integer >= len(p).
        If normalization is requested with n_cases <= 1.
    '''
    p = check_probabilities(p)
    
    if not np.isfinite(a) or a < 0:
        raise ValueError("a must be finite and non-negative.")

    # Number of possible outcomes
    if n_cases is None:
        n_cases = len(p)
        
    if n_cases % 1 != 0 or n_cases < len(p):
        raise ValueError("n_cases must be an integer >= len(p).")
    
    if norm and n_cases <= 1:
        raise ValueError("At least two possible outcomes are required for normalization.")

    mask = p>0
    p_masked = p[mask]
    
    if a == 1:
        # Shannon entropy limit
        Hr = entropy_shannon(p)
    
    else:
        Hr = 1/(1-a) * np.log(np.sum(p_masked**a))
    
    if norm:
        Hr = Hr / np.log(n_cases)

    return Hr


# %% Tsallis entropy

def entropy_tsallis(p, q, k=1, norm = False, n_cases = None): 
    '''
    Calculate Tsallis entropy.

    Parameters
    ----------
    p : Array
        Probabilities of the possible outcomes.
    q : float
        Non-negative entropic index based on Statistical Mechanical 
        arguments/denotes the divergence from extensivity.
    k : float, optional
        A positive scaling constant (e.g. Boltzmann constant k_B or k=1 in information theory). The default is 1.
    norm : bool, optional
        If True, normalize entropy (by its maximum value). The default is False.
    n_cases : int, optional
        Total number of possible outcomes. The default is len(p).


    Returns
    -------
    Ht : float
        Tsallis entropy (normalized if norm=True).

    Raises
    ------
    ValueError
        If p is not valid.
        If q or k is not finite, k is not positive, or q is negative.
        If n_cases is not an integer >= len(p).
        If normalization is requested with n_cases <= 1.
    '''
    
    p = check_probabilities(p)
    
    if not np.isfinite(q) or not np.isfinite(k):
        raise ValueError("q and k must be finite.")
    if k<=0 or q<0 :
        raise ValueError("k must be positive and q must be non-negative.")

    # Number of possible outcomes
    if n_cases is None:
        n_cases = len(p)
        
    if n_cases % 1 != 0 or n_cases < len(p):
        raise ValueError("n_cases must be an integer >= len(p).")
    
    if norm and n_cases <= 1:
        raise ValueError("At least two possible outcomes are required for normalization.")

    
    mask = p>0
    p_masked = p[mask]
    
    if q == 1:
        # Shannon entropy limit
        Ht = k * entropy_shannon(p)
        
        if norm:
            Hmax = k * np.log(n_cases)
            Ht = Ht / Hmax
            
    else:
        Ht = k * 1/(q-1) * (1-np.sum(p_masked**q))
    
        if norm:
            Hmax = k * 1/(q-1) * ( 1- n_cases**(1-q) )
            Ht = Ht / Hmax
        
    
    return Ht
   
    
    
    