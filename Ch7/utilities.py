import unittest
import warnings  # for suppressing numpy.array PendingDeprecationWarning's
import sys
sys.path.append("..")
from importlib import reload
import numpy as np
import sympy as sp
import control
import Ch6.utilities as ch6_utils
reload(ch6_utils)

def obsBassGura(A, C, desiredPoles, D=None, E=None, desiredExoPoles=None):
    """
    use Bass-Gura formalism to compute gain matrix, K, that places poles of the closed-loop metasystem \hat A- K \hat C
    at desired locations using algorithm outlined on Pg.272

    Inputs:
        A (numpy matrix/array, type=real) - system state matrix
        C (numpy matrix/array, type=real) - system measurement matrix
        desiredPoles (numpy matrix/array, type=complex) - desired pole locations
        D (numpy matrix/array, type=real) - (default=None) exogeneous measurement matrix
        E (numpy matrix/array, type=real) - (default=None) exogeneous input matrix 
        desiredExoPoles (numpy matrix/array, type=complex) - desired pole locations for the exogeneous observer
        
    Returns:
        (numpy matrix/array, type=real) - full observer gain matrix

    Raises:
        TypeError - if the optional inputs are improper
        ValueError - (reraises) if one of the called methods raises ValueError 
    """
    A = A.astype('float')
    C = C.astype('float')
    if all([inp is None for inp in (D, E, desiredExoPoles)]):
        try:
            return ch6_utils.bassGuraNoExo(A.T, C.T, desiredPoles).T
        except ValueError:
            raise ValueError('System is not observable')
    elif any([inp is None for inp in (D, E, desiredExoPoles)]):
        raise TypeError('D, E, desiredExoPoles must either all be None or all be numpy arrays')
    D = D.astype('float')
    E = E.astype('float')
    try:
        K = obsBassGura(A, C, desiredPoles)
    except ValueError:
        raise ValueError('Disturbance-free system is unobservable')
    try:
        K0 = obsBassGuraExo(A, C, K, D, E, desiredExoPoles)
    except NotImplementedError:
        K0 = np.zeros((E.shape[1], 1))
    return np.vstack([K, K0])

def obsBassGuraExo(A, C, K, D, E, desiredPoles):
    """
    compute observer gain, K0, for exogenous process estimate under the assumption that the disturbances are constant;
    see Section 7.4

    This method currently doesn't work.

    Inputs:
        A (numpy matrix/array, type=real) - system state matrix
        C (numpy matrix/array, type=real) - system measurement matrix
        K (numpy matrix/array, type=real) - disturbance-free observer gain matrix
        D (numpy matrix/array, type=real) - exogeneous input measurement matrix
        E (numpy matrix/array, type=real) - exogeneous input matrix
        desiredPoles (numpy matrix/array, type=complex) - desired pole locations (these are assumed unique!)
        
    Returns:
        (numpy matrix/array, type=real) - observer gain on exogeneous input

    Raises:
        NotImplementedError - the logic is currently not functional
    """
    raise NotImplementedError('The method in the book doesn\'t appear to work and there are no good examples to prove otherwise.')
    

def reducedOrderObserver(A11, A12, A21, A22, B1, B2, C1, desiredPoles):
    """
    compute matrices for the reduced order observer described in Section 7.5
    NOTE - the caller is expected to enforce the correct sizing of all matrices!

    Inputs:
        A11 (numpy matrix/array, type=real) - system state matrix corresponding to measurable subsystem change wrt measurable subsystem
        A12 (numpy matrix/array, type=real) - system state matrix corresponding to measurable subsystem change wrt unmeasurable subsystem
        A21 (numpy matrix/array, type=real) - system state matrix corresponding to unmeasurable subsystem change wrt measurable subsystem
        A22 (numpy matrix/array, type=real) - system state matrix corresponding to unmeasurable subsystem change wrt unmeasurable subsystem
        B1 (numpy matrix/array, type=real) - system control matrix corresponding to measurable subsystem change wrt control input
        B2 (numpy matrix/array, type=real) - system control matrix corresponding to unmeasurable subsystem change wrt control input
        C1 (numpy matrix/array, type=real) - system measurement matrix corresponding to measurable subsystem
        desiredPoles (numpy matrix/array, type=complex) - desired pole locations
        
    Returns:
        matrices L, Gbb, and H for the reduced order observer dynamics in Eqn. 7.54
        L - (numpy matrix/array, type=real) - reduced order observer gain matrix applied to estimated unmeasurable subsystem state
        Gbb - (numpy matrix/array, type=real) - reduced order observer gain matrix applied to system measurement y
        H - (numpy matrix/array, type=real) - reduced order observer gain matrix applied to control input u

    Raises:
        ValueError - if either (1) input matrices have improper size, or
                               (2) measurement matrix of available subsystem is not full-rank
    """
    if np.linalg.matrix_rank(C1) != C1.shape[0]:
        raise ValueError('Measurement matrix of available subsystem, C1, is not full-rank.')
    A11 = A11.astype('float')
    A12 = A12.astype('float')
    A21 = A21.astype('float')
    A22 = A22.astype('float')
    B1 = B1.astype('float')
    B2 = B2.astype('float')
    C1 = C1.astype('float')
    try:
        L = obsBassGura(A22, C1@A12, desiredPoles)
        Gbb = (A21 - L@(C1@A11))@np.linalg.inv(C1)
        H = B2 - L@(C1@B1)
        return L, Gbb, H
    except ValueError:
        raise ValueError('Are you sure all inputs have the correct shape?')

class TestCh7Utilities(unittest.TestCase):

    def test_obs_bass_gura(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # using constants from Problem 3.6
            m, M, l, g, k, R, r = .1, 1., 1., 9.8, 1., 100., .02
            A = np.array([[0., 1., 0., 0.], 
                          [0., -k**2 / (M*r**2*R), -(m*g)/M, 0.], 
                          [0., 0., 0., 1.], 
                          [0, k**2 / (M*r**2*R*l), ((M+m)*g)/(M*l), 0.]])
            C = np.array([[1., 0., 0., 0.]])
            B = np.array([[0.], [k/(M*R*r)], [0.], [-k/(M*R*r*l)]])
            w0 = 5.
            desiredPoles = np.roots(np.array([1.0, 2.613*w0, (2.+np.sqrt(2))*w0**2, 2.613*w0**3, w0**4]))
            K = obsBassGura(A, C, desiredPoles)
            # closed-loop system matrix
            Ac = A - K@C
            polesTest = ch6_utils.computePoles(Ac)
            # discount ordering by just checking against each entry of desiredPoles
            for p in polesTest:
                polePresent = any([np.isclose(np.real(p), np.real(pt)) and np.isclose(np.complex(p), np.complex(pt)) 
                    for pt in desiredPoles])
                self.assertTrue(polePresent)

    def test_reduced_order_observer(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # using constants from Problem 3.6
            m, M, l, g, k, R, r = .1, 1., 1., 9.8, 1., 100., .02
            A = np.array([[0., 1., 0., 0.], 
                          [0., -k**2 / (M*r**2*R), -(m*g)/M, 0.], 
                          [0., 0., 0., 1.], 
                          [0, k**2 / (M*r**2*R*l), ((M+m)*g)/(M*l), 0.]])
            B = np.array([[0.],[k/(M*R*r)],[0.],[-k/(M*R*r*l)]])
            A11 = np.array(A[0, 0]).reshape((1, 1))
            A12 = np.array(A[0, 1:]).reshape((1, 3))
            A21 = np.array(A[1:, 0]).reshape((3, 1))
            A22 = np.array(A[1:, 1:]).reshape((3, 3))
            B1 = np.array(B[0]).reshape((1, 1))
            B2 = np.array(B[1:]).reshape((3, 1))
            C1 = np.array([[1]]).reshape((1, 1))
            desiredPoles = np.roots(np.array([1.0, 2.*5., 2.*5.**2, 5.**3]))
            L, Gbb, H = reducedOrderObserver(A11, A12, A21, A22, B1, B2, C1, desiredPoles)
            # closed-loop system matrix
            F = A22 - L@(C1@A12)
            polesTest = ch6_utils.computePoles(F)
            # discount ordering by just checking against each entry of desiredPoles
            for p in polesTest:
                polePresent = any([np.isclose(np.real(p), np.real(pt)) and np.isclose(np.complex(p), np.complex(pt)) 
                    for pt in desiredPoles])
                self.assertTrue(polePresent)


if __name__ == '__main__':
    unittest.main()