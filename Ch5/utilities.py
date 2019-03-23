import numpy as np

def isControllable(A, B, computeSVD=False):
    """
    check if system is controllable

    NOTE(jwd) - the singular value decomposition gives insight to the most controllable directions of a system.

    Inputs:
        A (numpy matrix/array) - system state matrix
        B (numpy matrix/array) - system control matrix
        computeSVD (bool) - (default=False) compute singular value decomposition and return it
        
    Returns:
        controllable (bool) - True if the system meets controllability criteria, otherwise False
        singular value decomposition tuple (U, Sigma, V.T) if computeSVD=True, otherwise `None` 
    """
    A = A.astype('float')
    B = B.astype('float')
    rows = A.shape[0]
    nRows = rows
    nCols = rows * B.shape[1]
    # could also use control.ctrb()
    Q = np.empty([nRows, nCols])
    Q[:,:B.shape[1]] = B
    k = 1
    for i in range(B.shape[1], Q.shape[1], B.shape[1]):
        Ak = np.linalg.matrix_power(A, k)
        Q[:,i:i+B.shape[1]] = Ak@B
        k += 1
    rank = np.linalg.matrix_rank(Q)
    controllable = (rank == k)
    if computeSVD:
        U, Sigma, VT = np.linalg.svd(Q)
        return controllable, (U, Sigma, VT)
    return controllable, None

def isObservable(A, C, computeSVD=False):
    """
    check if system is observable.  This is dual to the controllability problem, so a call to this method is really
    the call `isControllable(A.T, C.T, one of {True, False})`

    NOTE(jwd) - the singular value decomposition gives insight to the most observable directions of a system.

    Inputs:
        A (numpy matrix/array) - system state matrix
        C (numpy matrix/array) - system measurement matrix
        computeSVD (bool) - (default=False) compute singular value decomposition and return it
        
    Returns:
        observable (bool) - True if the system meets observability criteria, otherwise False
        singular value decomposition tuple (U, Sigma, V.T) if computeSVD=True, otherwise `None` 
    """
    return isControllable(A.T, C.T, computeSVD=computeSVD)