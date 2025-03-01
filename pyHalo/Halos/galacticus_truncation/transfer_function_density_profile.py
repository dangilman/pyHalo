import warnings
import numpy as np
from scipy.special import hyp2f1
from scipy.integrate import quad
from scipy.interpolate import interp1d


# Implement the transfer function of subhalo density profile.

def _r_te_Du_2024(x, alpha=1.0, beta=3.0, gamma=1.0, delta=2.0):
    """
    Return the effective tidal (truncation) radius of a subhalo given the bound mass fraction.
    :param x: bound mass fraction
    :param alpha: alpha parameter in the generalized NFW profile
    :param beta: beta parameter in the generalized NFW profile
    :param gamma: gamma parameter in the generalized NFW profile
    :param delta: power index of the density truncation
    :return: the effective tidal (truncation) radius in units of infall virial radius
    """
    error_status = 0

    if (delta == 2.0):
        if (alpha == 1.0):
            if (beta == 3.0):
                if (gamma == 0.0):
                    A = -9.98429803e-01
                    B = 5.50290877e-04
                    C = 1.23471540e+00
                elif (gamma == 0.5):
                    A = -0.19210388
                    B = 0.42676044
                    C = 1.46250084
                elif (gamma == 1.0):
                    A = 0.68492777
                    B = 0.66438857
                    C = 2.07766512
                elif (gamma == 1.5):
                    A = 0.9839032
                    B = 0.76882264
                    C = 2.51841765
                else:
                    error_status = 1
            elif (beta == 4.0):
                if (gamma == 0.0):
                    A = 39.42152817
                    B = 0.58827911
                    C = 4.05363252
                elif (gamma == 0.5):
                    A = 2.06550731
                    B = 0.38713431
                    C = 3.63592324
                elif (gamma == 1.0):
                    A = 13.98955016
                    B = 0.72217703
                    C = 5.00880454
                elif (gamma == 1.5):
                    A = 83.39811168
                    B = 1.21616656
                    C = 6.62016776
                else:
                    error_status = 1
            else:
                error_status = 1
        elif (alpha == 2.0):
            if (beta == 3.0):
                if (gamma == 0.0):
                    A = -1.02415147e+00
                    B = 1.64345425e-03
                    C = 2.41297704e+00
                else:
                    error_status = 1
            elif (beta == 4.0):
                if (gamma == 0.0):
                    A = 3.27271231e-06
                    B = -1.94602678e+00
                    C = 5.84716369e+00
                else:
                    error_status = 1
                # throw a warning message if the x is too small, where the
                # fitting function might not work well.
                if (np.min(x) < 0.04):
                    warnings.warn(
                        'for [alpha=2, beta=4, gamma=0], the fitting function might not work reliably when the bound mass fraction is below 0.04')
            else:
                error_status = 1
        else:
            error_status = 1
    elif (delta == 3.0):
        if (alpha == 1.0):
            if (beta == 3.0):
                if (gamma == 0.0):
                    A = -0.93094144
                    B = 0.04703346
                    C = 0.76842857
                elif (gamma == 0.5):
                    A = -0.2290658
                    B = 0.41234666
                    C = 1.39936438
                elif (gamma == 1.0):
                    A = 0.90933265
                    B = 0.63676731
                    C = 2.18508243
                elif (gamma == 1.5):
                    A = 0.83533898
                    B = 0.73399035
                    C = 2.43215333
                else:
                    error_status = 1
            elif (beta == 4.0):
                if (gamma == 0.0):
                    A = 41.3266337
                    B = 0.60818104
                    C = 4.07045865
                elif (gamma == 0.5):
                    A = 0.42123475
                    B = -0.38164340
                    C = 3.62931001
                elif (gamma == 1.0):
                    A = 15.91886121
                    B = 0.71939954
                    C = 4.98186081
                elif (gamma == 1.5):
                    A = 104.0534991
                    B = 1.24696989
                    C = 6.62957129
                else:
                    error_status = 1
            else:
                error_status = 1
        elif (alpha == 2.0):
            if (beta == 3.0):
                if (gamma == 0.0):
                    A = -9.64566747e-01
                    B = -2.37053865e-03
                    C = 2.51094575e+00
                else:
                    error_status = 1
            elif (beta == 4.0):
                if (gamma == 0.0):
                    A = 5.79321402e-07
                    B = -1.62255246e+00
                    C = 5.47069835e+00
                else:
                    error_status = 1
                # throw a warning message if the x is too small, where the
                # fitting function might not work well.
                if (np.min(x) < 0.04):
                    warnings.warn(
                        'for [alpha=2, beta=4, gamma=0], the fitting function might not work reliably when the bound mass fraction is below 0.04')
            else:
                error_status = 1
        else:
            error_status = 1
    else:
        raise Exception('[delta] value not supported yet, set [delta] = 2 or 3')

    if (error_status):
        raise Exception('unsupported combination of [alpha, beta, gamma]')

    return (1.0 + A) * x ** B / (1.0 + A * x ** (2.0 * B)) / np.exp(C * (1.0 - x))


###############################################################################

def _f_t_Du_2024(x, alpha=1.0, beta=3.0, gamma=1.0, delta=2.0):
    """
    Return the density normalization of a subhalo given the bound mass fraction.
    :param x: bound mass fraction
    :param alpha: alpha parameter in the generalized NFW profile
    :param beta: beta parameter in the generalized NFW profile
    :param gamma: gamma parameter in the generalized NFW profile
    :param delta: power index of the density truncation
    :return: the density normalization
    """
    error_status = 0

    if (delta == 2.0):
        if (alpha == 1.0):
            if (beta == 3.0):
                if (gamma == 0.0):
                    D = 1.48242247
                    E = 0.63639230
                elif (gamma == 0.5):
                    D = 1.17122717
                    E = 0.45000174
                elif (gamma == 1.0):
                    D = 0.75826635
                    E = 0.23376409
                elif (gamma == 1.5):
                    D = -9.95103536e-01
                    E = 1.31454694e-04
                else:
                    error_status = 1
            elif (beta == 4.0):
                if (gamma == 0.0):
                    D = 1.18519061
                    E = 0.73729078
                elif (gamma == 0.5):
                    D = 0.75235790
                    E = 0.47771706
                elif (gamma == 1.0):
                    D = 0.36838147
                    E = 0.22307797
                elif (gamma == 1.5):
                    D = 0.28500656
                    E = 0.10528364
                else:
                    error_status = 1
            else:
                error_status = 1
        elif (alpha == 2.0):
            if (beta == 3.0):
                if (gamma == 0.0):
                    D = 1.37068893
                    E = 0.83149579
                else:
                    error_status = 1
            elif (beta == 4.0):
                if (gamma == 0.0):
                    D = 0.83836120
                    E = 1.23071929
                else:
                    error_status = 1
            else:
                error_status = 1
        else:
            error_status = 1
    elif (delta == 3.0):
        if (alpha == 1.0):
            if (beta == 3.0):
                if (gamma == 0.0):
                    D = 1.40209850
                    E = 0.63252747
                elif (gamma == 0.5):
                    D = 1.08663071
                    E = 0.45228951
                elif (gamma == 1.0):
                    D = 1.43637351
                    E = -0.24907289
                elif (gamma == 1.5):
                    D = 0.08092773
                    E = 0.08490512
                else:
                    error_status = 1
            elif (beta == 4.0):
                if (gamma == 0.0):
                    D = 1.08789526
                    E = 0.72799030
                elif (gamma == 0.5):
                    D = 0.68794863
                    E = 0.48619566
                elif (gamma == 1.0):
                    D = 0.33591171
                    E = 0.25084764
                elif (gamma == 1.5):
                    D = 0.15135479
                    E = 0.12838997
                else:
                    error_status = 1
            else:
                error_status = 1
        elif (alpha == 2.0):
            if (beta == 3.0):
                if (gamma == 0.0):
                    D = 1.31446961
                    E = 0.82402378
                else:
                    error_status = 1
            elif (beta == 4.0):
                if (gamma == 0.0):
                    D = 0.77798730
                    E = 1.21267992
                else:
                    error_status = 1
            else:
                error_status = 1
        else:
            error_status = 1
    else:
        raise Exception('[delta] value not supported yet, set [delta] = 2 or 3')

    if (error_status):
        raise Exception('unsupported combination of [alpha, beta, gamma]')

    # For x > 1, set f_t = 1.
    f_t = np.where(x <= 1.0, (1.0 + D) * x ** E / (1.0 + D * x ** (2.0 * E)), 1.0)

    # f_t should not be larger than 1.
    f_t = np.where(f_t <= 1.0, f_t, 1.0)

    return f_t


###############################################################################
#
# Note that the fittting functions above are derived in terms of the bound mass fraction
#
#   x = M_{bound} / M_{bound,0},
#
# and a fixed halo concentration of c_vir  = 20.6.
# For a different mass definition or halo concentration, the tracks of r_te(x) and f_t(x)
# might be different. So we redefine r_te and f_t in terms of the mass ratio
#
#   y = M_{bound} / M_{mx,0},
#
# where M_{mx,0} is the enclosed mass within the infall Rmax (the radius where the circular
# velocity reaches its maximum).
#
def _M_enclosed_generalized_NFW_dimensionless(x, alpha=1.0, beta=3.0, gamma=1.0):
    """
    Return the dimensionless enclosed mass within radius x = r / r_s for the generalized NFW profile.
    :param x: dimensionless radius
    :param alpha: alpha parameter in the generalized NFW profile
    :param beta: beta parameter in the generalized NFW profile
    :param gamma: gamma parameter in the generalized NFW profile
    :return: the dimensionless enclosed mass
    """
    status = 1

    # For special cases, use simple analytic expressions if available.
    if (alpha == 1.0):
        if (beta == 3.0):
            if (gamma == 0.0):
                mass = 4.0 \
                       * np.pi \
                       * ( \
                               +np.log(+1.0 + x) \
                               - x \
                               * (+2.0 + 3.0 * x) \
                               / 2.0 \
                               / (+1.0 + x) ** 2 \
                           )
            elif (gamma == 0.5):
                mass = +( \
                        -8.0 \
                        * np.pi \
                        * np.sqrt(x) \
                        * (3.0 + 4.0 * x) \
                    ) \
                       / ( \
                               +3.0 \
                               * (1.0 + x) ** 1.5 \
                           ) \
                       + 8.0 \
                       * np.pi \
                       * np.arcsinh(np.sqrt(x))
            elif (gamma == 1.0):
                mass = +4.0 \
                       * np.pi \
                       * ( \
                               +np.log(+1.0 + x) \
                               - x \
                               / (+1.0 + x) \
                           )
            elif (gamma == 1.5):
                mass = +8.0 \
                       * np.pi \
                       * ( \
                               -           np.sqrt(x / (1.0 + x)) \
                               + np.arcsinh(np.sqrt(x)) \
                           )
            else:
                status = 0
        elif (beta == 4.0):
            mass = 4.0 * np.pi / (3.0 - gamma) * (x / (1.0 + x)) ** (3.0 - gamma)
        else:
            status = 0
    elif (alpha == 2.0):
        if (beta == 3.0):
            if (gamma == 0.0):
                mass = -4.0 \
                       * np.pi \
                       * ( \
                               +x \
                               / np.sqrt(1.0 + x ** 2) \
                               + np.log(np.sqrt(1.0 + x ** 2) - x) \
                           )
            elif (gamma == 1.0):
                mass = 2.0 * np.pi * np.log(1.0 + x ** 2)
            else:
                status = 0
        elif (beta == 4.0):
            if (gamma == 0.0):
                mass = +2.0 \
                       * np.pi \
                       * ( \
                               +np.arctan(x) \
                               - x / (1.0 + x ** 2)
                       )
            elif (gamma == 1.0):
                mass = 4.0 * np.pi * (1.0 - 1.0 / np.sqrt(1.0 + x ** 2))
            else:
                status = 0
        else:
            status = 0
    else:
        status = 0

    if (status == 0):
        # Use the general expression.
        mass = +4.0 \
               * np.pi \
               * x ** (3.0 - gamma) \
               * hyp2f1((3.0 - gamma) / alpha, (beta - gamma) / alpha, 1.0 + (3.0 - gamma) / alpha, -x ** alpha) \
               / (3.0 - gamma)

    return mass


def _R_max_generalized_NFW_dimensionless(alpha=1.0, beta=3.0, gamma=1.0):
    """
    Return the dimensionless Rmax for the generalized NFW profile.
    :param x: dimensionless radius
    :param alpha: alpha parameter in the generalized NFW profile
    :param beta: beta parameter in the generalized NFW profile
    :param gamma: gamma parameter in the generalized NFW profile
    :return: the dimensionless Rmax
    """
    status = 1

    # Use expressions for special cases if available.
    if (alpha == 1.0):
        if (beta == 3.0):
            if (gamma < 2.0):
                if (gamma == 0.0):
                    Rmax = 4.424700646872270
                elif (gamma == 0.5):
                    Rmax = 3.289276561384110
                elif (gamma == 1.0):
                    Rmax = 2.162581587064612
                elif (gamma == 1.5):
                    Rmax = 1.054966571869124
                else:
                    status = 0
            else:
                raise Exception('Rmax does not exist')
        elif (beta == 4.0):
            if (gamma < 2.0):
                Rmax = 2.0 - gamma
            else:
                raise Exception('Rmax does not exist')
        else:
            status = 0
    elif (alpha == 2.0):
        if (beta == 3.0):
            if (gamma < 2.0):
                if (gamma == 0.0):
                    Rmx = 2.919847688299723
                else:
                    status = 0
            else:
                raise Exception('Rmax does not exist')
        elif (beta == 4.0):
            if (gamma < 2.0):
                if (gamma == 0.0):
                    Rmx = 1.825255642519694
                else:
                    status = 0
            else:
                raise Exception('Rmax does not exist')
        else:
            status = 0
    else:
        status = 0

    if (status == 0):
        # Use numerical solver.
        def root(x):
            return (1.0 + x ** alpha) ** ((gamma - beta) / alpha) \
                   - hyp2f1((3.0 - gamma) / alpha, (beta - gamma) / alpha, 1.0 + (3.0 - gamma) / alpha, -x ** alpha) \
                   / (3.0 - gamma)

        Rmax = newton(root, 1.0)

    return Rmax


def _M_total_generalized_NFW_Truncated_dimensionless(tau, alpha=1.0, beta=3.0, gamma=1.0, delta=2.0):
    """
    Return the dimensionless total mass of a subhalo with truncated generalized NFW profile:
    rho(x) = rho_org(x) / (1.0 + (x / tau)**2),
    where rho_org(x) is the original density profile (without truncation), x = r / r_s, and tau = r_te / r_s.
    :param tau: dimensionless truncation radius
    :param alpha: alpha parameter in the generalized NFW profile
    :param beta: beta parameter in the generalized NFW profile
    :param gamma: gamma parameter in the generalized NFW profile
    :param delta: power index of the density truncation
    :return: the dimensionless enclosed mass
    """
    status = 1

    if (delta == 2.0):
        if (alpha == 1.0):
            if (beta == 3.0):
                if (gamma == 0.0):
                    mass = 4.0 * np.pi \
                           * tau ** 2 / (tau ** 2 + 1.0) ** 3 \
                           * ( \
                                   0.5 * (-1.0 + (np.pi - tau) * tau) * (3.0 * tau ** 2 - 1.0) \
                                   + tau ** 2 * (tau ** 2 - 3.0) * np.log(tau) \
                               )
                elif (gamma == 0.5):
                    atan_tau = np.arctan(tau)
                    mass = 4.0 * np.pi \
                           * tau ** 2 / (tau ** 2 + 1.0) ** (9.0 / 4.0) \
                           / 6.0 \
                           * ( \
                                   3.0 * np.sqrt(2.0) * np.pi * np.sqrt(tau) * (tau ** 2 + 1.0) \
                                   * (np.sin(2.5 * atan_tau) - np.cos(2.5 * atan_tau)) \
                                   + 2.0 * np.sqrt(2.0) \
                                   * (np.sqrt(1.0 + np.sqrt(2.0 * tau) + tau) - np.sqrt(1.0 - np.sqrt(2.0 * tau) + tau)) \
                                   * np.sqrt(tau * (tau ** 2 + 1.0) * (+1.0 - tau + np.sqrt(tau ** 2 + 1.0))) \
                                   * np.cos(0.25 * (np.pi + 2.0 * atan_tau)) \
                                   - 2.0 * np.sqrt(2.0) \
                                   * (np.sqrt(1.0 + np.sqrt(2.0 * tau) + tau) + np.sqrt(1.0 - np.sqrt(2.0 * tau) + tau)) \
                                   * np.sqrt(tau * (tau ** 2 + 1.0) * (-1.0 + tau + np.sqrt(tau ** 2 + 1.0))) \
                                   * np.sin(0.25 * (np.pi + 2.0 * atan_tau)) \
                                   + ( \
                                           +( \
                                                   +2.0 * np.sqrt(2.0) + 4.0 * np.sqrt(tau) + 3.0 * np.sqrt(2.0) * tau \
                                                   - 2.0 * np.sqrt(2.0) * tau ** 2 - 4.0 * tau ** 2.5 - np.sqrt(
                                                   2.0) * tau ** 3 \
                                               ) * np.sqrt(1.0 - np.sqrt(2.0 * tau) + tau) \
                                           + ( \
                                                   +2.0 * np.sqrt(2.0) - 4.0 * np.sqrt(tau) + 3.0 * np.sqrt(2.0) * tau \
                                                   - 2.0 * np.sqrt(2.0) * tau ** 2 + 4.0 * tau ** 2.5 - np.sqrt(
                                                   2.0) * tau ** 3 \
                                               ) * np.sqrt(1.0 + np.sqrt(2.0 * tau) + tau) \
                                       ) \
                                   * np.sqrt((-1.0 + tau + np.sqrt(tau ** 2 + 1.0)) / tau) \
                                   * np.cos(0.25 * (np.pi + 2.0 * atan_tau)) \
                                   + ( \
                                           +( \
                                                   +2.0 * np.sqrt(2.0) + 4.0 * np.sqrt(tau) + 3.0 * np.sqrt(2.0) * tau \
                                                   - 2.0 * np.sqrt(2.0) * tau ** 2 - 4.0 * tau ** 2.5 - np.sqrt(
                                                   2.0) * tau ** 3 \
                                               ) * np.sqrt(1.0 - np.sqrt(2.0 * tau) + tau) \
                                           - ( \
                                                   +2.0 * np.sqrt(2.0) - 4.0 * np.sqrt(tau) + 3.0 * np.sqrt(2.0) * tau \
                                                   - 2.0 * np.sqrt(2.0) * tau ** 2 + 4.0 * tau ** 2.5 - np.sqrt(
                                                   2.0) * tau ** 3 \
                                               ) * np.sqrt(1.0 + np.sqrt(2.0 * tau) + tau) \
                                       ) \
                                   * np.sqrt((+1.0 - tau + np.sqrt(tau ** 2 + 1.0)) / tau) \
                                   * np.sin(0.25 * (np.pi + 2.0 * atan_tau)) \
                                   - 12.0 \
                                   * ( \
                                           +(tau ** 2 - 1.0) * np.arcsin(
                                           np.sqrt(0.5 * (1.0 + tau - np.sqrt(tau ** 2 + 1.0)))) \
                                           + tau \
                                           * np.log(tau + np.sqrt(tau ** 2 + 1.0) + np.sqrt(
                                           2.0 * tau * (tau + np.sqrt(tau ** 2 + 1.0)))) \
                                       ) \
                                   * np.sqrt(tau) \
                                   * np.cos(0.25 * (np.pi + 2.0 * atan_tau)) \
                                   - 12.0 \
                                   * ( \
                                           +2.0 * tau * np.arcsin(np.sqrt(0.5 * (1.0 + tau - np.sqrt(tau ** 2 + 1.0)))) \
                                           - 0.5 * (tau ** 2 - 1.0) \
                                           * np.log(tau + np.sqrt(tau ** 2 + 1.0) + np.sqrt(
                                           2.0 * tau * (tau + np.sqrt(tau ** 2 + 1.0)))) \
                                       ) \
                                   * np.sqrt(tau) \
                                   * np.sin(0.25 * (np.pi + 2.0 * atan_tau)) \
                               )
                elif (gamma == 1.0):
                    mass = 4.0 * np.pi \
                           * tau ** 2 / (tau ** 2 + 1.0) ** 2 \
                           * ( \
                                   (tau ** 2 - 1.0) * np.log(tau) \
                                   + tau * np.pi \
                                   - (tau ** 2 + 1.0)
                           )
                elif (gamma == 1.5):
                    atan_tau = np.arctan(tau)
                    mass = 4.0 * np.pi \
                           * tau ** 1.5 / (tau ** 2 + 1.0) ** (5.0 / 4.0) \
                           / 2.0 \
                           * ( \
                                   +np.sqrt(2.0) * np.pi * np.sqrt(tau ** 2 + 1.0) \
                                   * (np.cos(1.5 * atan_tau) + np.sin(1.5 * atan_tau)) \
                                   - np.sqrt(2.0) \
                                   * ( \
                                           +np.sqrt(1.0 + np.sqrt(2.0 * tau) + tau) \
                                           + np.sqrt(1.0 - np.sqrt(2.0 * tau) + tau) \
                                       ) \
                                   * np.sqrt((tau ** 2 + 1.0) * (-1.0 + tau + np.sqrt(tau ** 2 + 1.0))) \
                                   * np.cos(0.25 * (np.pi + 2.0 * atan_tau)) \
                                   - np.sqrt(2.0) \
                                   * ( \
                                           +np.sqrt(1.0 + np.sqrt(2.0 * tau) + tau) \
                                           - np.sqrt(1.0 - np.sqrt(2.0 * tau) + tau) \
                                       ) \
                                   * np.sqrt((tau ** 2 + 1.0) * (+1.0 - tau + np.sqrt(tau ** 2 + 1.0))) \
                                   * np.sin(0.25 * (np.pi + 2.0 * atan_tau)) \
                                   - 2.0 \
                                   * ( \
                                           +2.0 * tau * np.arcsin(np.sqrt(0.5 * (1.0 + tau - np.sqrt(tau ** 2 + 1.0)))) \
                                           + np.log(tau + np.sqrt(tau ** 2 + 1.0) + np.sqrt(
                                           2.0 * tau * (tau + np.sqrt(tau ** 2 + 1.0)))) \
                                       ) \
                                   * np.cos(0.25 * (np.pi + 2.0 * atan_tau)) \
                                   - 2.0 \
                                   * ( \
                                           +2.0 * np.arcsin(np.sqrt(0.5 * (1.0 + tau - np.sqrt(tau ** 2 + 1.0)))) \
                                           - tau \
                                           * np.log(tau + np.sqrt(tau ** 2 + 1.0) + np.sqrt(
                                           2.0 * tau * (tau + np.sqrt(tau ** 2 + 1.0)))) \
                                       ) \
                                   * np.sin(0.25 * (np.pi + 2.0 * atan_tau)) \
                               )
                else:
                    status = 0
            else:
                status = 0
        elif (alpha == 2.0):
            if (beta == 4.0):
                if (gamma == 0.0):
                    mass = np.pi ** 2 * tau ** 2 / (tau + 1.0) ** 2
                else:
                    status = 0
            else:
                status = 0
        else:
            status = 0
    elif (delta == 3.0):
        if (alpha == 1.0 and beta == 3.0 and gamma == 1.0):
            mass = 4.0 * np.pi \
                   * tau ** 2 / (tau ** 3 - 1.0) ** 2 \
                   * ( \
                           -tau * (tau ** 3 - 1.0) \
                           + 2.0 * np.sqrt(3.0) * np.pi / 9.0 \
                           * (tau - 1.0) ** 2 \
                           * (2.0 * tau + 1.0) \
                           + tau \
                           * (tau ** 3 + 2.0) \
                           * np.log(tau) \
                       )
        else:
            status = 0
    else:
        status = 0

    if (status == 0):
        # Use numerical integration.
        def integral(x):
            return 4.0 * np.pi * x ** 2 \
                   / x ** gamma \
                   / (1.0 + x ** alpha) ** ((beta - gamma) / alpha) \
                   / (1.0 + (x / tau) ** delta)

        mass = quad(integral, 0.0, np.inf)

    return mass


# Mass table used to compute the radius enclosing a certain mass for truncated generalized NFW profile.
N_per_decade = 10
x_min = 1.0e-6
x_max = 1.0e+4
N_tab = int(np.log10(x_max / x_min) * N_per_decade)
x_tab = np.geomspace(x_min, x_max, N_tab)
M_tab = np.zeros(N_tab)

Mass_table_Initialized = False
alpha_used = None
beta_used = None
gamma_used = None
delta_used = None
M_min = np.inf
M_max = 0.0


def _Truncation_Radius(M, f_t, alpha=1.0, beta=3.0, gamma=1.0, delta=2.0):
    """
    Return the truncation radius of a subhalo given the bound mass and normalization of density profile.
    The density profile is assumed to be
    rho(x) = rho_org(x) * f_t / (1.0 + (x / tau)**2),
    where rho_org(x) is the original density profile (without truncation), x = r / r_s, and tau = r_te / r_s.
    :param M: dimensionless mass of the subhalo
    :param f_t: normalization of the density profile
    :param alpha: alpha parameter in the generalized NFW profile
    :param beta: beta parameter in the generalized NFW profile
    :param gamma: gamma parameter in the generalized NFW profile
    :param delta: power index of the density truncation
    :return: the dimensionless truncation radius
    """

    global N_per_decade, x_min, x_max, N_tab, x_tab, M_tab
    global Mass_table_Initialized, alpha_used, beta_used, gamma_used, delta_used, M_min, M_max

    M_unnormalized = M / f_t

    retabulate = True
    if (Mass_table_Initialized and alpha_used == alpha and beta_used == beta and gamma_used == gamma \
        and delta_used == delta and np.min(M_unnormalized) >= M_min and np.max(M_unnormalized) <= M_max):
        retabulate = False
    if (retabulate):
        M_min = _M_total_generalized_NFW_Truncated_dimensionless(x_min, alpha, beta, gamma, delta)
        M_max = _M_total_generalized_NFW_Truncated_dimensionless(x_max, alpha, beta, gamma, delta)
        while (np.min(M_unnormalized) < M_min):
            x_min = x_min / 2.0
            M_min = _M_total_generalized_NFW_Truncated_dimensionless(x_min, alpha, beta, gamma, delta)

        while (np.max(M_unnormalized) > M_max):
            x_max = x_max * 2.0
            M_max = _M_total_generalized_NFW_Truncated_dimensionless(x_max, alpha, beta, gamma, delta)

        N_tab = int(np.log10(x_max / x_min) * N_per_decade)
        x_tab = np.geomspace(x_min, x_max, N_tab)
        M_tab = _M_total_generalized_NFW_Truncated_dimensionless(x_tab, alpha, beta, gamma, delta)

        Mass_table_Initialized = True
        alpha_used = alpha
        beta_used = beta
        gamma_used = gamma
        delta_used = delta

    x_M_interp_loglog = interp1d(np.log(M_tab), np.log(x_tab), kind='cubic')

    return np.exp(x_M_interp_loglog(np.log(M_unnormalized)))


def Convert_to_reference_model(alpha=1.0, beta=3.0, gamma=1.0):
    """
    Compute the conversion factors from parameter y = M_{bound} / M_{mx,0} to parameter
    x = M_{bound} / M_{bound,0} in the reference model.
    :param alpha: alpha parameter in the generalized NFW profile
    :param beta: beta parameter in the generalized NFW profile
    :param gamma: gamma parameter in the generalized NFW profile
    :return: conversion factors [Mmx_ref / Mbound_ref_0, Rmax_ref / Rvir_ref]
    """
    error_status = 0

    if (alpha == 1.0):
        if (beta == 3.0):
            if (gamma == 0.0):
                Mbound_ref_0 = 1.1969934e9  # in units of Msun
            elif (gamma == 0.5):
                Mbound_ref_0 = 1.1793903e9  # in units of Msun
            elif (gamma == 1.0):
                Mbound_ref_0 = 1.1605002e9  # in units of Msun
            elif (gamma == 1.5):
                Mbound_ref_0 = 1.1395515e9  # in units of Msun
            else:
                error_status = 1
        elif (beta == 4.0):
            if (gamma == 0.0):
                Mbound_ref_0 = 1.0426536e9  # in units of Msun
            elif (gamma == 0.5):
                Mbound_ref_0 = 1.0353760e9  # in units of Msun
            elif (gamma == 1.0):
                Mbound_ref_0 = 1.0281669e9  # in units of Msun
            elif (gamma == 1.5):
                Mbound_ref_0 = 1.0210254e9  # in units of Msun
            else:
                error_status = 1
        else:
            error_status = 1
    elif (alpha == 2.0):
        if (beta == 3.0):
            if (gamma == 0.0):
                Mbound_ref_0 = 1.1343262e9  # in units of Msun
            else:
                error_status = 1
        elif (beta == 4.0):
            if (gamma == 0.0):
                Mbound_ref_0 = 1.0194281e9  # in units of Msun
            else:
                error_status = 1
        else:
            error_status = 1
    else:
        error_status = 1

    if (error_status):
        raise Exception('unsupported combination of [alpha, beta, gamma]')

    Mvir_ref = 1.0e9  # in units of Msun, note that Mbound_ref_0 is slightly larger than Mvir_ref
    Rvir_ref = 0.02632439488393568  # in units of Mpc
    c_ref = 20.587820037799197
    Rmax_dimensionless = _R_max_generalized_NFW_dimensionless(alpha, beta, gamma)
    Rmax_ref = Rmax_dimensionless / c_ref * Rvir_ref
    Mmx_ref = _M_enclosed_generalized_NFW_dimensionless(Rmax_dimensionless, alpha, beta, gamma) \
              / _M_enclosed_generalized_NFW_dimensionless(c_ref, alpha, beta, gamma) \
              * Mvir_ref

    return Mmx_ref / Mbound_ref_0, Rmax_ref / Rvir_ref


def compute_r_te_and_f_t(Mbound, MvirInfall, RvirInfall, cInfall, alpha=1.0, beta=3.0, gamma=1.0, delta=2.0):
    """
    Return the effective tidal (truncation) radius and the density normalization of a subhalo given the
    bound mass and infall halo properties.
    :param Mbound: bound mass
    :param MvirInfall: virial mass at infall
    :param RvirInfall: virial radius at infall
    :param cInfall: halo concentration at infall
    :param alpha: alpha parameter in the generalized NFW profile
    :param beta: beta parameter in the generalized NFW profile
    :param gamma: gamma parameter in the generalized NFW profile
    :param delta: power index of the density truncation
    :return: the effective tidal (truncation) radius
    """

    Rs = RvirInfall / cInfall
    Rmax_dimensionless = _R_max_generalized_NFW_dimensionless(alpha, beta, gamma)

    # Characteristic mass.
    M0 = MvirInfall / _M_enclosed_generalized_NFW_dimensionless(cInfall, alpha, beta, gamma)

    M_mx = M0 * _M_enclosed_generalized_NFW_dimensionless(Rmax_dimensionless, alpha, beta, gamma)

    y = Mbound / M_mx

    # Convert y to parameter x = MBoud / MvirInfall in the reference model.
    y_scale, R_scale = Convert_to_reference_model(alpha, beta, gamma)

    # r_te = r_te_Du_2024(y * y_scale, alpha, beta, gamma, delta) / R_scale * Rmax
    ## Make sure that the truncation radius is smaller than the virial radius at infall.
    # r_te = np.where(r_te < RvirInfall, r_te, RvirInfall)

    f_t = _f_t_Du_2024(y * y_scale, alpha, beta, gamma, delta)

    # Instead of directly using the fitting function for r_te, we first compute f_t
    # using the fitting function and then compute r_te assuming that the total mass
    # of the truncated halo equals to the bound mass. r_te computed in this way is
    # in good agreement with the fitting function, but is more reliable where the
    # fitting function is not calibrated to simulations.
    Mbound_dimensionless = Mbound / M0
    r_te = Rs * _Truncation_Radius(Mbound_dimensionless, f_t, alpha, beta, gamma)

    return r_te, f_t
