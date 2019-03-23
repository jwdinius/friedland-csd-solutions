import numpy as np
import sympy as sp
import control

s, w = sp.symbols('s w')

def computeResolvent(A, imag=False, smplfy=True):
    """
    compute resolvent of a square matrix (see Eqn 3.49)

    Inputs:
        A (numpy matrix/array) - real square matrix
        imag (bool) - (default=False) use s=i*w for computation
        smplfy (bool) - (default=True) do partial fraction decomposition on resolvent

    Returns:
        resolvent, (sI-A)**(-1), Eqn. 3.49 in the book
    """
    assert(A.shape[0] == A.shape[1])
    nRows = A.shape[0]  # == nCols
    if imag:
        res = ((sp.I*w)*sp.eye(nRows) - A)**-1
    else:
        res = (s*sp.eye(nRows) - A)**-1
    if not smplfy:
        return res
    # perform partial fraction decomposition term-by-term
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            # apart does partial fraction decomp automatically
            res[i,j] = sp.apart(sp.simplify(res[i,j]), s)
    return res

def firstCompanionForm(num, den):
    """
    compute first companion form given single-input single-output (SISO) transfer function representation
    see Eqns. 3.88-3.94

    Inputs:
        num (sympy Poly) - transfer function numerator
        den (sympy Poly) - transfer function denominator

    Returns:
        sympy matrices A, B, C, D with
        x_dot = Ax + Bu
        y = Cx + Du
        for the First Companion form discussed in the book
    """
    #single-input, single-output
    # H(s) = num / den
    # convert to H(s)
    # den = s^k + a1*s^(k-1) + a2*s^(k-2)+...
    a = den.coeffs()
    # num = b0*s^k + b1*s^(k-1)+...
    b = num.coeffs()
    # append the coefficients array if there is no constant term in the numerator
    if sp.degree(num) > len(num.coeffs())-1:
        b.append(0)
    if sp.degree(den) > len(den.coeffs())-1:
        a.append(0)
    # if the denominator has higher order than the numerator, prepend 0's for the leading coeffs until
    # a,b have the same size
    if len(a) > len(b):
        # prepend b
        for i in range(len(a)-len(b)):
            b.insert(0,0)
    # construct A, B matrices (Eqn 3.88)
    A = sp.zeros(len(a)-1)
    for i in range(A.shape[0]-1):
        A[i, i+1] = 1
    # coefficients order is reversed (w.r.t. the book's convention) by sp, so reverse it to 
    # match the book's convention
    a.reverse()
    b.reverse()
    for i in range(A.shape[0]):
        A[-1, i] = -a[i]
    B = sp.zeros(A.shape[0], 1)
    B[-1] = 1
    # construct C,D matrices (Eqn 3.94) 
    C = sp.zeros(1, A.shape[1])
    for i in range(C.shape[1]):
        C[0, i] = b[i] - a[i]*b[-1]
    D = sp.Matrix([b[-1]])
    return A, B, C, D


def jordanForm(num, den, D=sp.Matrix([0])):
    """
    compute partial fraction decomposition Jordan form given single-input single-output (SISO) transfer function representation
    see Eqns. 3.108, 3.111, and 3.116

    Inputs:
        num (sympy Poly) - transfer function numerator
        den (sympy Poly) - transfer function denominator
        D (sympy Matrix) - (default=Matrix([0]) direct path from input u to output y

    Returns:
        sympy matrices A, B, C, D with
        x_dot = Ax + Bu
        y = Cx + Du
        for the Jordan form discussed in the book

    Raises:
        NotImplementedError - raised when one of the following three conditions is encountered:
                                (1) repeated roots
                                (2) order(numer of decomposed system) > 1
                                (3) order(denom of decomposed system) > 2
    """
    uniqueRoots = np.unique(np.array(den.all_roots()).astype('complex'))
    order = den.degree()
    if uniqueRoots.size < order:
        raise NotImplementedError('Method for repeated roots is not implemented.')
    num = sp.factor(num)
    den = sp.factor(den)
    pd = sp.apart(num/den)
    A = sp.zeros(order, order)
    B = sp.zeros(order, 1)
    C = sp.zeros(1, order)
    idx = 0
    for i, p in enumerate(pd.args):
        n, d = sp.fraction(p)
        # extract multiplicative factor from the denomintor
        # - the desired form for each term in the partial fraction decomposition is a / (s + r), with a, r some real numbers
        _d = sp.factor(sp.Poly(d.as_expr(), s))
        multFactor = _d.func(*[term for term in _d.args if not term.free_symbols])
        # numerator
        numerCoeffs = sp.Poly(n.as_expr(), s).all_coeffs()
        if len(numerCoeffs) == 1:
            # constant poly
            b, a = numerCoeffs[0] / multFactor, 0
        elif len(numerCoeffs) == 2:
            # linear poly
            b, a = [_n / multFactor for _n in numerCoeffs]
        else:
            raise NotImplementedError('Order of numerator {num} is too large for this method.'.format(num=sp.Poly(n.as_expr(), s)))
        # denominator
        denPoly = sp.Poly(d.as_expr(), s)
        if denPoly.degree() == 2:
            # this will only happen there are complex conjugate pairs
            cq, bq, aq = denPoly.all_coeffs()
            # need to set a subsystem here (see Eqn. 3.111)
            twoTimesSigma, sigmaSqPlusOmegaSq = bq / aq, cq / aq
            twoTimesLambda, twoTimesCrossProd = b, a
            A[idx, idx+1] = 1
            A[idx+1, idx] = -sigmaSqPlusOmegaSq
            A[idx+1, idx+1] = -twoTimesSigma
            B[idx+1] = 1
            C[0, idx] = twoTimesCrossProd
            C[0, idx+1] = twoTimesLambda
            idx += 2
        elif denPoly.degree() == 1:
            poles = sp.polys.polyroots.roots_linear(denPoly)
            A[idx, idx] = poles[0]
            B[idx] = 1
            C[0, idx] = b
            idx += 1
        else:
            NotImplementedError('Order of denominator {den} is too large for this method.'.format(den=denPoly))
    return A, B, C, D