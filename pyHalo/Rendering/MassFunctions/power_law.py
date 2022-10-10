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
                 mass_function_model_class, kwargs_suppression_function):


        self.draw_poisson = draw_poisson
        self._index = power_law_index
        self._mL = 10 ** log_mlow
        self._mH = 10 ** log_mhigh

        self._mass_function_model_class = mass_function_model_class
        self._kwargs_suppression_function = kwargs_suppression_function

        self._nhalos_mean_unbroken = self._mass_function_model_class.integrate_power_law_analytic(normalization,
                                                                                             10 ** log_mlow,
                                                                                             10 ** log_mhigh, 0,
                                                                                             power_law_index)

    def draw(self):

        """
        Draws samples from a double power law distribution between mL and mH of the form
        m ^ power_law_index * (1 + (a*mc / m)^b )^c

        Physically, the second term multiplying m^power_law_index can be a suppression in the mass function on small
        scales.

        :param draw_poisson:
        :param _index:
        :param _mH:
        :param _mL:
        :param n_draw:
        :return:
        """

        m = self._sample(self.draw_poisson, self._index, self._mH, self._mL, self._nhalos_mean_unbroken)

        if len(m) == 0:
            return m

        factor = self._mass_function_model_class.suppression(m, **self._kwargs_suppression_function)
        u = np.random.rand(int(len(m)))
        inds = np.where(u < factor)
        return m[inds]

    def _sample(self, draw_poisson, index, mH, mL, n_draw):

        """
        Draws samples from a power law distribution between mL and mH
        :param draw_poisson:
        :param _index:
        :param _mH:
        :param _mL:
        :param n_draw:
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
