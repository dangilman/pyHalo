import numpy as np


class GeneralPowerLaw(object):

    """
    This class handles computations of a double power law mass function of the form
    dn/dm = m^x * (1 + (a * m_c / m)^b)^c
    where a, b, and c are constants, and m_c is a characteristic mass scale.

    The keywords for a, b, c are a_wdm, b_wdm, and c_wdm, respectively

    Lovell 2020 fit this mass function to simulations of Warm Dark Matter cosmologies and find
    (a, b, c) = (2.3, 0.8, -1) for central halos and (4.2, 2.5, -0.2) for subhalos
    """

    def __init__(self, log_mlow, log_mhigh, power_law_index, draw_poisson, normalization,
                 suppression_model_class, kwargs_suppression_function):
        """

        :param log_mlow: log10(minimum halo mass)
        :param log_mhigh: log10(maximum halo mass)
        :param power_law_index: logarithmic slope of the halo mass function between mlow and mhigh
        :param draw_poisson: bool; whether to draw number of halos from a poisson distribution
        :param normalization: the amplitude of the mass function
        :param suppression_model_class: a function that returns the suppression of the mass function as a function of
        halo mass (for example, for warm dark matter)
        :param kwargs_suppression_function: keyword arguments for calls to suppresion_model_class.suppression
        """

        self.draw_poisson = draw_poisson
        self._index = power_law_index
        self._mL = 10 ** log_mlow
        self._mH = 10 ** log_mhigh

        self._suppression_model_class = suppression_model_class
        self._kwargs_suppression_function = kwargs_suppression_function

        self._nhalos_mean_unbroken = self._suppression_model_class.integrate_power_law_analytic(normalization,
                                                                                             10 ** log_mlow,
                                                                                             10 ** log_mhigh, 0,
                                                                                             power_law_index)

    def draw(self):

        """
        Generate samples from the mass function
        """

        m = self._sample(self.draw_poisson, self._index, self._mH, self._mL, self._nhalos_mean_unbroken)

        if len(m) == 0:
            return m

        factor = self._suppression_model_class.suppression(m, **self._kwargs_suppression_function)

        u = np.random.rand(int(len(m)))
        inds = np.where(u < factor)
        return m[inds]

    def _sample(self, draw_poisson, index, mH, mL, n_draw):

        """
        Samples from the mass function
        :param draw_poisson: bool; generate samples from poisson distribution
        :param index: logarithmic slope
        :param mH: high mass
        :param mL: low mass
        :param n_draw: number of halos to draw
        :return:
        """

        if draw_poisson:
            N = np.random.poisson(n_draw)
        else:
            N = int(round(np.round(n_draw)))

        x = np.random.rand(N)
        if index == -1:
            norm = np.log(mH / mL)
            X = mL * np.exp(norm * x)
        else:
            X = (x * (mH ** (1 + index) - mL ** (1 + index)) + mL ** (
                1 + index)) ** (
                    (1 + index) ** -1)

        return np.array(X)
