import numpy as np
from pyHalo.Rendering.MassFunctions.util import integrate_power_law_analytic, integrate_power_law_quad
from scipy.interpolate import interp1d


class _PowerLawBase(object):

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
        x = np.random.rand(ndraw)
        if self._index == -1:
            norm = np.log(mH / mL)
            X = mL * np.exp(norm * x)
        else:
            X = (x * (mH ** (1 + self._index) - mL ** (1 + self._index)) + mL ** (
                1 + self._index)) ** (
                    (1 + self._index) ** -1)
        return np.array(X)

class _PowerLawTurnoverBase(object):

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
        print(self.n_mean, self.first_moment)

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
       :param mixed_DM_frac: fraction of dark matter in WDM component
       """
        kwargs_mass_function = {'log_mc': log_mc, 'a_mfunc_break': a_wdm, 'b_mfunc_break': b_wdm,
                                'c_mfunc_break': c_wdm, 'mixed_DM_frac': mixed_DM_frac}
        print(power_law_index)
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

# mfunc = PowerLawTurnover(6.0, 10.0, 7.5, -1.9, False, 10**12, 1.0, 1.0, -2.0)
# m = mfunc.draw()
# import matplotlib.pyplot as plt
# h, b = np.histogram(np.log10(m), range=(6, 10), bins=20)
# plt.plot(b[0:-1], np.log10(h))
# plt.show()
# print(mfunc.n_mean)
# print(len(m))
# print(np.sum(m)/mfunc.first_moment)
# class Tabulated(object):
#
#     """
#     Samples from an arbitrary mass function by numerically inverting the CDF
#     """
#     def __init__(self, log_mlow, log_mhigh, log_m_evaluated, log_massfunc_amplitude, draw_poisson):
#
#         self._logmlow = log_mlow
#         self._logmhigh = log_mhigh
#         m = 10 ** log_m_evaluated
#         mfunc = 10 ** log_m_evaluated
#         cdf = np.cumsum(mfunc)
#         cdf_inverse = interp1d(cdf, m)
#
# norm = 10 ** 9
# _logm = np.linspace(6, 10, 10000)
# m = 10 ** _logm
# dn_dm = norm * (m) ** -1.9
# n_m = []
# for i in range(np.min(m), np.max(m), 20):
#     cond1 =
#     n_m.append(np)
#
#
# mfunc_differential = np.absolute(np.gradient(norm * (m) ** -1.9, m))
# log_mfunc_differential = np.log10(_mfunc_differential)
#
# differential_mfunc = 10 ** log_mfunc_differential
# differential_mfunc_pdf = differential_mfunc / np.sum(differential_mfunc)
# cdf = np.cumsum(differential_mfunc_pdf)
# u_min = cdf[0]
# u_max = cdf[-1]
# print(cdf)
# cdf_inverse = interp1d(cdf, m)
# m_sampled = cdf_inverse(np.random.uniform(u_min, u_max, 1000))
#
# import matplotlib.pyplot as plt
# h, x = np.histogram(np.log10(m_sampled), range=(6, 10), bins=20)
#
# plt.plot(x[0:-1], np.log10(h), color='r')
# mfunc_plaw = PowerLaw(6.0, 10.0, -1.9, False, norm)
# m_sampled_theory = mfunc_plaw.draw()
# htheory, x_theory = np.histogram(np.log10(m_sampled_theory), range=(6, 10), bins=20)
# plt.plot(x_theory[0:-1], np.log10(htheory), color='k')
# plt.show()
# print(np.polyfit(np.log10(htheory), np.log10(x_theory[0:-1]),1))
