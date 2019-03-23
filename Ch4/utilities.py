import sympy as sp

def hurwitz(Dcoeffs):
    """
    compute determinants for hurwitz stability criteria (see Section 4.5)

    Inputs:
        Dcoeffs - coefficients of denominator polynomial ordered from highest order to lowest

    Returns:
        the determinants to be used for evaluating stability via the Hurwitz criterion
    """
    assert(isinstance(Dcoeffs, list))
    # rescale D's coefficients so that coefficient on s^k is 1, where k is the order of the denominator
    rescaledCoeffs = [d / Dcoeffs[0] for d in Dcoeffs]
    # zero pad
    order = len(rescaledCoeffs)-1
    coeffsPadded = sp.zeros(2*order,1)
    for i in range(order+1):
        coeffsPadded[i] = rescaledCoeffs[i]
    hurwitzDeterminants = []
    for i in range(order):
        # construct subminor matrix
        Co = sp.zeros(i+1,i+1)
        skip = 0
        for j in range(i+1):
            if j > 0 and j % 2 == 0:
                skip += 1
            if j % 2 == 0:
                # put odd coefficients (a1, a3, a5...)
                k = skip
                ii = 0
                while k <= i:
                    Co[j,k] = coeffsPadded[2*ii+1]
                    ii += 1
                    k += 1
            else:
                # put even coefficients (a0=1, a2, a4...)
                k = skip
                ii = 0
                while k <= i:
                    Co[j,k] = coeffsPadded[2*ii]
                    ii += 1
                    k += 1
        Do = sp.simplify(sp.det(Co))
        hurwitzDeterminants.append(Do)
    return hurwitzDeterminants
