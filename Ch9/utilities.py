from importlib import reload
import sys
sys.path.append("..")
import numpy as np
import Ch6.utilities as ch6_utils
reload(ch6_utils)
import Ch7.utilities as ch7_utils
reload(ch7_utils)
import control

def initialResponse(A, B, C, D, x0, startTime, finalTime, dt, G=None):
    """
    compute transient response of a continuous-time linear system (A, B, C, D)

    Inputs:
        A (numpy matrix/array, type=real) - system state matrix
        B (numpy matrix/array, type=real) - system control matrix
        C (numpy matrix/array, type=real) - system measurement matrix
        D (numpy matrix/array, type=real) - direct path from input to measurement
        x0 (numpy matrix/array, type=real) - initial state
        startTime (float) - initial time
        finalTime (float) - final time
        dt (float) - timestep
        G (numpy matrix/array, type=real) - (default=None) feedback gain to apply; if default is not overriden,
                                            G will be an appropriately-sized zero matrix.

    Returns:
        T (numpy matrix/array, type=real) - times
        y (numpy matrix/array, type=real) - simulated system measurements at times T

    """
    def f(x, t, u, A, B):
        """ function to integrate """
        return A@x + B@u
    
    def rk4Step(x, t, u, A, B, dt):
        """ runge-kutta 4th order integration scheme"""
        k1 = dt * f(x, t, u, A, B)
        k2 = dt * f(x + k1/2., t + dt/2., u, A, B)
        k3 = dt * f(x + k2/2., t + dt/2., u, A, B)
        k4 = dt * f(x + k3, t + dt, u, A, B)
        return x + (k1 + 2.*k2 + 2.*k3 + k4) / 6.
    if G is None:
        G = np.zeros((B.shape[1], A.shape[0]))    
        
    nTimes = int((finalTime - startTime) / dt) + 1
    T = np.linspace(startTime, finalTime, nTimes)
    _x0 = x0.reshape((len(x0),1))
    y = np.empty((len(T), C.shape[0]))
    y[0, :] = (C@_x0).reshape((1, C.shape[0]))
    for i in range(1, len(T)):
        u = -G@_x0
        _x = rk4Step(_x0, T[i-1], u, A, B, dt)
        _y = C@_x
        y[i, :] = np.array(_y).reshape((1, C.shape[0]))
        _x0 = _x
    return T, y

def exogeneousGains(A, B, C, D, E, Q, R, method="bass-gura"):
    """
    compute gain matrix on exogeneous input using either the Bass-Gura method from Chapter 6 or the
    method described by Eqns. 9.49 and 9.50

    Inputs:
        A (numpy matrix/array, type=real) - system state matrix
        B (numpy matrix/array, type=real) - system control matrix
        C (numpy matrix/array, type=real) - system measurement matrix
        D (numpy matrix/array, type=real) - direct path from input to measurement
        E (numpy matrix/array, type=real) - exogeneous input matrix
        Q (numpy matrix/array, type=real) - LQR regulation penalty
        R (numpy matrix/array, type=real) - LQR control penalty
        method (string) - (default='bass-gura') - trigger on method to use

    Returns:
        (numpy matrix/array, type=real)- exogeneous gain matrices using Bass-Gura method or LQR method

    Raises:
        ValueError - if the system is unobservable
    """
    A = A.astype('float')
    B = B.astype('float')
    C = C.astype('float')
    D = D.astype('float')
    E = E.astype('float')
    Q = Q.astype('float')
    R = R.astype('float')
    if np.linalg.matrix_rank(ch6_utils.observabilityMatrix(A, C)) != A.shape[0]:
        raise ValueError('System is unobservable')
    G, S, _ = control.lqr(A, B, Q, R)
    if method == "bass-gura":
        return ch6_utils.bassGuraExo(A, B, C, G.astype('float'), E)
    # implement Eqns. 9.49 and 9.50
    Ac = A - B@G
    Bstar = -np.linalg.inv(R)@(B.T)@np.linalg.inv(Ac.T)@S
    return Bstar@E