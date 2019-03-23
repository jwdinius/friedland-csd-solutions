import unittest
import warnings  # for suppressing numpy.array PendingDeprecationWarning's
from importlib import reload
import sys
sys.path.append("..")
import numpy as np
import sympy as sp
import Ch6.utilities as ch6_utils
import Ch7.utilities as ch7_utils
reload(ch7_utils)
import control

def fullOrderCompensator(A, B, C, D, controlPoles, observerPoles):
    """
    combine controller and observer constructions into a compensator design
    currently, supports single-input-single-output systems only

    Inputs:
        A (numpy matrix/array, type=real) - system state matrix
        B (numpy matrix/array, type=real) - system control matrix
        C (numpy matrix/array, type=real) - system measurement matrix
        desiredPoles (numpy matrix/array, type=complex) - desired pole locations
        D (numpy matrix/array, type=real) - direct path from control to measurement matrix
        E (numpy matrix/array, type=real) - (default=None) exogeneous input matrix 
        controlPoles (numpy matrix/array, type=complex) - desired controller system poles
        observerPoles (numpy matrix/array, type=complex) - desired observer system poles

    Returns:
        (control.TransferFunction) - compensator transfer function from input to output

    Raises:
        TypeError - if input system is not SISO
    """
    if B.shape[1] > 1 or C.shape[0] > 1:
        raise TypeError('Only single-input-single-output (SISO) systems are currently supported')
    A = A.astype('float')
    B = B.astype('float')
    C = C.astype('float')
    D = D.astype('float')
    G = ch6_utils.bassGura(A, B, controlPoles)
    K = ch7_utils.obsBassGura(A, C, observerPoles)
    return control.ss2tf(control.StateSpace(A - B@G - K@C, K, G, D))

def stabilityRange(tfD, tfPlant, gain):
    """
    numerically evaluate a range of gains, k, to see if the closed-loop system (k*tfD*tfPlant) / (1 + k*tfD*tfPlant)

    Inputs:
        tfD (control.TransferFunction) - compensator transfer function
        tfPlant (control.TransferFunction) - open loop system transfer function
        gain (numpy matrix/array, type=real) - range of gains to check

    Returns:
        tuple(min gain, max gain) defining stability interval if such an interval could be found, `None` otherwise

    Raises:
    """
    stable = np.zeros_like(gain)
    for i,_k in enumerate(gain):
        rz = np.real(control.zero(1.+(_k*tfD)*tfPlant))
        #print(rz.size)
        if rz.size > 0 and np.max(rz) < 0:
            stable[i] = 1.
    if len(gain[stable > 0]) > 1:
        return (gain[stable > 0][0], gain[stable > 0][-1])
    return None


class TestCh8Utilities(unittest.TestCase):

    def test_full_order_compensator(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            A = np.array([[0., 1., 0., 0.], [0., -25., -0.98, 0.], [0., 0., 0., 1.], [0., 25., 10.78, 0.]]).astype('float')
            B = np.array([[0.], [0.5], [0.], [-0.5]]).astype('float')
            C = np.array([[1., 0., 0., 0.]]).astype('float')
            D = np.zeros((1, 1)).astype('float')
            desiredPolesCtrl = np.array([-25., -4., np.complex(-2., 2.*np.sqrt(3)), np.complex(-2., -2.*np.sqrt(3))])
            w0 = 5.
            desiredPolesObsv = np.roots(np.array([1.0, 2.613*w0, (2.+np.sqrt(2))*w0**2, 2.613*w0**3, w0**4]))
            compensatorTF = fullOrderCompensator(A, B, C, D, desiredPolesCtrl, desiredPolesObsv)
            openLoopTF = control.ss2tf(control.StateSpace(A, B, C, D))
            # poles are zeros of return difference
            poles = control.zero(1.+compensatorTF*openLoopTF)
            for p in poles:
                polePresent = any([np.isclose(np.complex(p), np.complex(pt)) for pt in np.hstack([desiredPolesCtrl, desiredPolesObsv])])
                self.assertTrue(polePresent)

    def test_reduced_order_compensator(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            A = np.array([[0., 1., 0., 0.], [0., -25., -0.98, 0.], [0., 0., 0., 1.], [0., 25., 10.78, 0.]]).astype('float')
            B = np.array([[0.], [0.5], [0.], [-0.5]]).reshape((4,1)).astype('float')
            C = np.array([[1., 0., 0., 0.]]).reshape((1, 4)).astype('float')
            D = np.zeros((1, 1)).astype('float')

            ### Controller ###
            desiredPolesCtrl = np.array([-25., -4., np.complex(-2., 2.*np.sqrt(3)), np.complex(-2., -2.*np.sqrt(3))])
            G = ch6_utils.bassGura(A, B, desiredPolesCtrl)
            G1 = G[0, 0].reshape((1, 1))
            G2 = G[0, 1:].reshape((1, 3))
            
            ### Observer ###
            desiredPolesObsv = np.roots(np.array([1.0, 2.*5., 2.*5.**2, 5.**3]))
            Aaa = np.array(A[0, 0]).reshape((1, 1))
            Aau = np.array(A[0, 1:]).reshape((1, 3))
            Aua = np.array(A[1:, 0]).reshape((3, 1))
            Auu = np.array(A[1:, 1:]).reshape((3, 3))
            Ba = np.array(B[0]).reshape((1, 1))
            Bu = np.array(B[1:]).reshape((3, 1))
            Ca = np.array([[1]]).reshape((1, 1)).astype('float')
            L, Gbb, H = ch7_utils.reducedOrderObserver(Aaa, Aau, Aua, Auu, Ba, Bu, Ca, desiredPolesObsv)

            ### Compensator ###
            F = Auu - L@Aau
            Ar = F-H@G2
            Br = Ar@L + Gbb - H@G1
            Cr = G2
            Dr = G1+G2@L
            sysD = control.StateSpace(Ar, Br, Cr, Dr)
            compensatorTF = control.ss2tf(sysD)
            sysplant = control.StateSpace(A, B, C, D)
            openLoopTF = control.ss2tf(sysplant)
            # poles are zeros of return difference
            poles = control.zero(1.+compensatorTF*openLoopTF)
            for p in poles:
                polePresent = any([np.isclose(np.complex(p), np.complex(pt)) for pt in np.hstack([desiredPolesCtrl, desiredPolesObsv])])
                self.assertTrue(polePresent)

if __name__ == '__main__':
    unittest.main()