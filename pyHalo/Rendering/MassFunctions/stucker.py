import numpy as np
from scipy.optimize import root

def stucker_suppression_params(dlogT_dlogk, gamma=5.0):

    """
    Maps from the logarithmic derivative of the transfer function at the half-mode scale to the a, b, c parameters
    that describe the suppression of the halo mass function.

    Credit for these routines goes to Jens Stucker: https://arxiv.org/pdf/2109.09760.pdf
    Please cite the paper when using this routine
    :param dlogT_dlogk: the logarithmic derivative of the transfer function at k half-mode
    :return: the a, b, c parameters for the mass function suppression
    """

    fabg = (-1.0 + 0.5 ** (-1 / gamma))
    beta = dlogT_dlogk * (1 + fabg) / fabg / gamma

    if (beta > 6.) | (beta < 1.5):
        print("beta = %.2f is outside the validated range [1.5 ... 6]!"
                      " Usage at your own risk!" % beta)
    mu = np.array((0.2651, 1.638, 16.51))
    nu = np.array((0.3656, -0.0994, -0.9466))
    (m20, m50, m80) = 1.0*mu*beta**nu
    a_stucker, b, c = mscales_to_abc(m20, m50, m80)
    a = a_stucker ** b
    return a, b, c

def mscales_to_abc(m20, m50, m80):
    """Uniquely maps the parameters m20, m50, m80 to a, b, c
    Credit for these routines goes to Jens Stucker: https://arxiv.org/pdf/2109.09760.pdf
    Please cite the paper when using this routine
    """
    def _equations(par):
        a, b, c =  10**par[0], par[1], par[2]
        ms = _supp_mscale(a, b, c, frac=np.array([0.2,0.5,0.8]))
        eq = abs(np.log10(ms) - np.log10([m20, m50, m80]))/0.01
        return eq
    p0 = np.array([np.log10(m50), 2., -1.])
    res = root(_equations, p0, method="lm")
    return 10**res.x[0], res.x[1], res.x[2]

def _supp_mscale(a, b, c, frac=0.5):
    """The mass scale where the surpression reaches a given value, given a,b,c
    a, b, c : parameters (a >= 0, b >= 0, c <= 0)
    frac : desired surpression factor
    returns : the mass M where f(M)=frac, units are as the parameter "a"

    Credit for these routines goes to Jens Stucker: https://arxiv.org/pdf/2109.09760.pdf
    Please cite the paper when using this routine
    """
    if (a < 0) | (b < 0) | (c > 0):
        # This is just here, to tell the fitting function when it goes out of bounds
        return np.nan
    return a / (frac**(1./c) - 1.)**(1./b)
