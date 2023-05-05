import numpy as np
from pyHalo.Rendering.MassFunctions.util import integrate_power_law_analytic, integrate_power_law_quad
from scipy.interpolate import interp1d

__all__ = ['CDMPowerLaw', 'WDMPowerLaw', 'MixedWDMPowerLaw']

class _PowerLawBase(object):
    name = 'BASE_POWER_LAW'
    def __init__(self, log_mlow, log_mhigh, power_law_index, draw_poisson, normalization):
        """
        :param log_mlow: log10(minimum halo mass)
        :param log_mhigh: log10(maximum halo mass)
        :param power_law_index: logarithmic slope of the halo mass function between mlow and mhigh
        :param draw_poisson: bool; whether to draw number of halos from a poisson distribution
        :param normalization: the amplitude of the mass function
        """
        self._logmlow = log_mlow
        self._logmhigh = log_mhigh
        self._index = power_law_index
        self._draw_poisson = draw_poisson
        self._normalization = normalization
        self.n_mean = integrate_power_law_analytic(normalization, 10 ** log_mlow, 10**log_mhigh,
                                                   0.0, power_law_index)
        self.first_moment = integrate_power_law_analytic(normalization, 10 ** log_mlow, 10**log_mhigh,
                                                   1.0, power_law_index)

    def draw(self):

        """
        Sample from the mass function
        """

        mH = 10 ** self._logmhigh
        mL = 10 ** self._logmlow
        if self._draw_poisson:
            ndraw = np.random.poisson(self.n_mean)
        else:
            ndraw = int(round(np.round(self.n_mean)))
        x = np.random.uniform(0, 1, ndraw)
        if self._index == -1:
            norm = np.log(mH / mL)
            X = mL * np.exp(norm * x)
        else:
            X = (x * (mH ** (1 + self._index) - mL ** (1 + self._index)) + mL ** (
                1 + self._index)) ** (
                    (1 + self._index) ** -1)
        return np.array(X)

class _PowerLawTurnoverBase(object):
    name = 'BASE_POWER_LAW_TURNOVER'
    def __init__(self, log_mlow, log_mhigh, power_law_index, draw_poisson,
                 normalization, kwargs_mass_function):
        """

        :param log_mlow: log10(minimum halo mass)
        :param log_mhigh: log10(maximum halo mass)
        :param log_mc: log10(halo mass) where turnover occurs
        :param power_law_index: logarithmic slope of the halo mass function between mlow and mhigh
        :param draw_poisson: bool; whether to draw number of halos from a poisson distribution
        :param normalization: the amplitude of the mass function
        :param a_mfunc_break: coefficient of turnover
        :param b_mfunc_break: exponent of turnover
        :param c_mfunc_break: outere exponent of turnover
        """
        self._logmlow = log_mlow
        self._logmhigh = log_mhigh
        self._index = power_law_index
        self._draw_poisson = draw_poisson
        self._normalization = normalization
        self._kwargs_mass_function = kwargs_mass_function
        self._mass_function_unbroken = _PowerLawBase(log_mlow, log_mhigh, power_law_index, draw_poisson, normalization)
        self.n_mean = integrate_power_law_quad(normalization, 10 ** log_mlow, 10 ** log_mhigh,
                                                         0.0, power_law_index, self._turnover,
                                                     self._kwargs_mass_function)

        self.first_moment = integrate_power_law_quad(normalization, 10 ** log_mlow, 10 ** log_mhigh,
                                                         1.0, power_law_index, self._turnover,
                                                     self._kwargs_mass_function)

    def draw(self):
        """
        Samples from the mass function
        :return:
        """
        m = self._mass_function_unbroken.draw()
        if len(m) == 0:
            return m
        factor = self._turnover(m, **self._kwargs_mass_function)
        u = np.random.rand(int(len(m)))
        inds = np.where(u < factor)
        return m[inds]

    @staticmethod
    def _turnover(*args, **kwargs):
        raise Exception('must specify turnover method')


class CDMPowerLaw(_PowerLawBase):
    name = 'CDM_POWER_LAW'
    def __init__(self, log_mlow, log_mhigh, power_law_index, draw_poisson, normalization, *args, **kwargs):
        """

       :param log_mlow: log10(minimum halo mass)
       :param log_mhigh: log10(maximum halo mass)
       :param power_law_index: logarithmic slope of the halo mass function between mlow and mhigh
       :param draw_poisson: bool; whether to draw number of halos from a poisson distribution
       :param normalization: the amplitude of the mass function
       """
        super(CDMPowerLaw, self).__init__(log_mlow, log_mhigh, power_law_index, draw_poisson, normalization)


class WDMPowerLaw(_PowerLawTurnoverBase):
    name = 'WDM_POWER_LAW'
    def __init__(self, log_mlow, log_mhigh, power_law_index, draw_poisson, normalization,
                 log_mc, a_wdm, b_wdm, c_wdm, *args, **kwargs):
        """

       :param log_mlow: log10(minimum halo mass)
       :param log_mhigh: log10(maximum halo mass)
       :param power_law_index: logarithmic slope of the halo mass function between mlow and mhigh
       :param draw_poisson: bool; whether to draw number of halos from a poisson distribution
       :param normalization: the amplitude of the mass function
       :param log_mc: log10(halo mass) where turnover occurs
       :param a_wdm: coefficient of turnover
       :param b_wdm: exponent of turnover
       :param c_wdm: outer exponent of turnover
       """

        kwargs_mass_function = {'log_mc': log_mc, 'a_mfunc_break': a_wdm, 'b_mfunc_break': b_wdm,
                                'c_mfunc_break': c_wdm}
        super(WDMPowerLaw, self).__init__(log_mlow, log_mhigh, power_law_index, draw_poisson, normalization,
                                          kwargs_mass_function)

    def draw(self):
        """
        Samples from the mass function
        :return:
        """
        m = self._mass_function_unbroken.draw()
        if len(m) == 0:
            return m
        factor = self._turnover(m, **self._kwargs_mass_function)
        u = np.random.rand(int(len(m)))
        inds = np.where(u < factor)
        return m[inds]

    @staticmethod
    def _turnover(m, log_mc, a_mfunc_break, b_mfunc_break, c_mfunc_break):
        """

        :param m:
        :param log_mc:
        :param a_mfunc_break:
        :param b_mfunc_break:
        :param c_mfunc_break:
        :return:
        """
        m_c = 10 ** log_mc
        r = a_mfunc_break * (m_c / m) ** b_mfunc_break
        factor = 1 + r
        return factor ** c_mfunc_break

class MixedWDMPowerLaw(_PowerLawTurnoverBase):
    name = 'MIXED_DM_POWER_LAW'
    def __init__(self, log_mlow, log_mhigh, power_law_index, draw_poisson, normalization,
                 log_mc, a_wdm, b_wdm, c_wdm, mixed_DM_frac, *args, **kwargs):
        """

       :param log_mlow: log10(minimum halo mass)
       :param log_mhigh: log10(maximum halo mass)
       :param power_law_index: logarithmic slope of the halo mass function between mlow and mhigh
       :param draw_poisson: bool; whether to draw number of halos from a poisson distribution
       :param normalization: the amplitude of the mass function
       :param log_mc: log10(halo mass) where turnover occurs
       :param a_wdm: coefficient of turnover
       :param b_wdm: exponent of turnover
       :param c_wdm: outer exponent of turnover
       :param mixed_DM_frac: fraction of dark matter in CDM component
       """
        kwargs_mass_function = {'log_mc': log_mc, 'a_mfunc_break': a_wdm, 'b_mfunc_break': b_wdm,
                                'c_mfunc_break': c_wdm, 'mixed_DM_frac': mixed_DM_frac}
        super(MixedWDMPowerLaw, self).__init__(log_mlow, log_mhigh, power_law_index, draw_poisson,
                                               normalization, kwargs_mass_function)

    @staticmethod
    def _turnover(m, log_mc, a_mfunc_break, b_mfunc_break, c_mfunc_break, mixed_DM_frac):
        """

        :param m:
        :param log_mc:
        :param a_mfunc_break:
        :param b_mfunc_break:
        :param c_mfunc_break:
        :param mixed_DM_frac:
        :return:
        """
        m_c = 10 ** log_mc
        r = a_mfunc_break * (m_c / m) ** b_mfunc_break
        factor = 1 + r
        wdm_comp = factor ** c_mfunc_break
        suppression_factor = (mixed_DM_frac + (1 - mixed_DM_frac) * np.sqrt(wdm_comp)) ** 2
        return suppression_factor
