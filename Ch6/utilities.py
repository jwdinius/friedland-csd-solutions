import unittest
import warnings  # for suppressing numpy.array PendingDeprecationWarning's
import numpy as np
from control import ctrb as controllabilityMatrix

def observabilityMatrix(A, C):
    """
    compute observability matrix corresponding to state matrix A and measurement matrix C

    Inputs:
        A (numpy matrix/array, type=real) - system state matrix
        C (numpy matrix/array, type=real) - system measurement matrix

    Returns:
        (numpy matrix/array, type=real) - observability matrix
    """
    # the observability problem is dual to the controllability problem, so use a common method for both
    return controllabilityMatrix(A.T, C.T)

def bassGura(A, B, desiredPoles, C=None, E=None):
    """
    use Bass-Gura formalism to compute gain matrix, G, that places poles of the closed-loop meta-system \hat A- \hat B*G
    at desired locations

    Inputs:
        A (numpy matrix/array, type=real) - system state matrix
        B (numpy matrix/array, type=real) - system control matrix
        desiredPoles (numpy matrix/array, type=complex) - desired pole locations
        C (numpy matrix/array, type=real) - (default=None) system measurement matrix
        E (numpy matrix/array, type=real) - (default=None) exogeneous input matrix 
        
    Returns:
        (numpy matrix/array, type=real) - full gain matrix

    Raises:
        TypeError - if the input matrices C and E are improper
        ValueError - (reraises) if one of the called methods raises ValueError 
    """
    A = A.astype('float')
    B = B.astype('float')
    if all([inp is None for inp in (C, E)]):
        return bassGuraNoExo(A, B, desiredPoles)
    elif (C is None and not E is None) or (E is None and not C is None):
        raise TypeError('C, E must either both be None or both be numpy arrays')
    C = C.astype('float')
    E = E.astype('float')
    G = bassGuraNoExo(A, B, desiredPoles)
    G0 = bassGuraExo(A, B, C, G, E)
    return np.hstack([G, G0])


def bassGuraNoExo(A, B, desiredPoles):
    """
    use Bass-Gura formalism to compute gain matrix G that places poles of the closed-loop system A-B*G
    at desired locations

    Inputs:
        A (numpy matrix/array, type=real) - system state matrix
        B (numpy matrix/array, type=real) - system control matrix
        desiredPoles (numpy matrix/array, type=complex) - desired pole locations
        
    Returns:
        (numpy matrix/array, type=real) - gain matrix to apply to undisturbed state

    Raises:
        ValueError - if the system is uncontrollable 
    """
    desiredPoles = np.ravel(desiredPoles)
    openLoopCoeffs = computeCoeffs(computePoles(A))
    closedLoopCoeffs = computeCoeffs(desiredPoles)
    Q = controllabilityMatrix(A, B).astype('float')
    if np.linalg.matrix_rank(Q) != A.shape[0]:
        raise ValueError('System is not controllable')
    W = np.eye(A.shape[0]).astype('float')
    for i in range(W.shape[0]):
        W[i, i+1:] = openLoopCoeffs[1:-(i+1)]
    deltaCoeffs = closedLoopCoeffs - openLoopCoeffs
    return np.linalg.inv((Q@W).T)@deltaCoeffs[1:]

def bassGuraExo(A, B, C, G, E):
    """
    compute exogenous gain matrix G0 given Bass-Gura gain matrix, G, and system matrices A, B, E

    Inputs:
        A (numpy matrix/array, type=real) - system state matrix
        B (numpy matrix/array, type=real) - system control matrix
        C (numpy matrix/array, type=real) - system measurement matrix
        G (numpy matrix/array, type=real) - state gain
        E (numpy matrix/array, type=real) - exogeneous input matrix

    Returns:
        (numpy matrix/array, type=real)- exogenous gain matrix

    Raises:
        ValueError - if the system is unobservable
    """
    A = A.astype('float')
    B = B.astype('float')
    C = C.astype('float')
    G = G.astype('float')
    E = E.astype('float')
    if np.linalg.matrix_rank(observabilityMatrix(A, C)) != A.shape[0]:
        raise ValueError('System is unobservable')
    Ac = A - B@G
    AcInv = np.linalg.inv(Ac)
    CAcInv = C@AcInv
    CAcInvB = (C@AcInv)@B
    Bpound = np.linalg.pinv(CAcInvB)@CAcInv
    return Bpound@E

def computePoles(A):
    """
    compute roots of the characteristic equation of a state matrix

    Inputs:
        A (numpy matrix/array, type=real) - system state matrix

    Returns:
        eigenvalues (flattened numpy array, type=complex) - eigenvalues of matrix A
    """
    eigenvalues, _ = np.linalg.eig(A.astype('float'))
    return np.ravel(eigenvalues.astype('complex'))

def computeCoeffs(roots):
    """
    compute coefficients of a characteristic polynomial with given roots

    Inputs:
        roots (flattened numpy array, type=complex) - roots of a polynomial

    Returns:
         (flattened numpy array, type=complex) - polynomial coefficients
    """
    roots = roots.astype('complex')
    return np.ravel(np.poly(roots))


class TestCh6Utilities(unittest.TestCase):
    
    def test_bass_gura_no_exo(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m, M, l, g, k, R, r = .1, 1., 1., 9.8, 1., 100., .02
            A = np.array([[0., 1., 0., 0.],
                [0., -k**2 / (M*r**2*R), -(m*g)/M, 0.],
                [0., 0., 0., 1.],
                [0, k**2 / (M*r**2*R*l), ((M+m)*g)/(M*l), 0.]]).astype('float')
            B = np.array([[0.],[k/(M*R*r)],[0.],[-k/(M*R*r*l)]]).astype('float')
            C = np.array([[1., 0., 0., 0.]]).astype('float')
            D = np.array([[0]]).astype('float')
            desiredPoles = np.array([-25., -4., np.complex(-2., 2.*np.sqrt(3)), np.complex(-2., -2.*np.sqrt(3))]).astype('complex')
            G = bassGuraNoExo(A, B, desiredPoles)
            # closed-loop system matrix
            Ac = A - B@G
            polesTest = computePoles(Ac)
            # discount ordering by just checking against each entry of desiredPoles (slow, but no big deal here)
            for p in polesTest:
                polePresent = any([np.isclose(np.real(p), np.real(pt)) and np.isclose(np.complex(p), np.complex(pt)) 
                    for pt in desiredPoles])
                self.assertTrue(polePresent)

    def test_bass_gura(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            YbOverV = -.746
            YpOverV = .006
            YrOverV = .001
            gOverV = .0369
            YaOverV = .0012
            YRoverV = .0092
            Lb = -12.9
            Lp = -.746
            Lr = .387
            La = 6.05
            LR = .952
            Nb = 4.31
            Np = .024
            Nr = -.174
            Na = -.416
            NR = -1.76
            T = 0.2
            A = np.array([[-LR*(YpOverV / YRoverV + Lp), -LR*((YrOverV - 1.) / YRoverV + Lr), -LR*gOverV / YRoverV],
                        [-NR*(YpOverV / YRoverV + Np), -NR*((YrOverV - 1.) / YRoverV + Nr), -NR*gOverV / YRoverV],
                        [1., 0., 0.]])
            B = np.array([[-LR*(YaOverV / YRoverV + La)],
                        [-NR*(YaOverV / YRoverV + Na)],
                        [0.]])
            C = np.array([[0, 0, 1]])
            E = np.array([[-LR*(T*YbOverV + 1.) / (T*YRoverV)],
                        [-NR*(T*YbOverV + 1.) / (T*YRoverV)],
                        [0.]])
            desiredPoles = np.array([-1., np.complex(-1., 3.), np.complex(-1., -3.)])
            G = bassGura(A, B, desiredPoles, C, E)

            # now the check: setup the metasystem and check the closed-loop poles
            Ameta = np.zeros((4, 4))
            Bmeta = np.zeros((4, 1))
            Ameta[:3, :3] = A
            Ameta[:3, -1] = E.ravel()
            Ameta[-1, -1] = -1./T
            Bmeta[:3] = B
            # combine the gain matrices
            AmetaCL = Ameta - Bmeta@G
            polesTest = computePoles(AmetaCL)
            # append the pole at -1/T
            desiredMetaPoles = np.hstack([desiredPoles, -T**(-1)]).ravel()
            # discount ordering by just checking against each entry of desiredPoles (slow, but no big deal here)
            for p in polesTest:
                polePresent = any([np.isclose(np.real(p), np.real(pt)) and np.isclose(np.complex(p), np.complex(pt)) 
                    for pt in desiredMetaPoles])
                self.assertTrue(polePresent)


if __name__ == '__main__':
    unittest.main()