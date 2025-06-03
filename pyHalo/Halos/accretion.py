import numpy
from scipy.interpolate import interp1d
from scipy.stats import truncnorm


class InfallDistributionGalacticus2024(object):
    """ACCRETION REDSHIFT PDF FROM GALACTICUS USING THE VERSION OF GALACTICUS AS OF FEB 2024 AND SELECTING ON
    SUBHALOS WITH A BOUND MASS > 10^6"""
    name = 'version2024'
    def __init__(self, z_lens):
        self._z_lens = z_lens
        self._counts = numpy.array([ 50,  93, 125, 180, 175, 144, 120, 117,  97,  82,  52,  51,  35,
        20,   9,   4,   4,   0,   1,   1])
        self._z_infall = numpy.array([ 0.53836189,  1.37376234,  2.2091628 ,  3.04456325,  3.87996371,
        4.71536416,  5.55076461,  6.38616507,  7.22156552,  8.05696598,
        8.89236643,  9.72776689, 10.56316734, 11.39856779, 12.23396825,
       13.0693687 , 13.90476916, 14.74016961, 15.57557007, 16.41097052]) - 0.5
        cdf = numpy.cumsum(self._counts)
        self._cdf = cdf / numpy.max(cdf)
        self._cdf_min = numpy.min(self._cdf)
        self._cdf_max = numpy.max(self._cdf)
        self._interp = interp1d(self._cdf, self._z_infall)

    def __call__(self, *args, **kwargs):
        u = numpy.random.uniform(self._cdf_min, self._cdf_max)
        z_infall = self._z_lens + self._interp(u)
        return z_infall

class InfallDistributionHybrid(object):
    name = 'hybrid'
    """Accretion redshift pdf that is a combination of directly infall and indirectly infall halos"""
    def __init__(self, z_lens, log_m_host):
        """

        :param z_lens: main deflector redshift
        :param log_m_host: log10 host halo mass in solar masses
        """
        self._m_host = 10**log_m_host
        self._z_lens = z_lens

    def __call__(self, m_sub):
        """
        Return the infall redshift for a subhalo with infall mass m_sub
        :param m_sub: infall mass in solar masses
        :return: infall redshift
        """
        mass_ratio = self._mass_ratio_in_bounds(m_sub / self._m_host)
        mu = self.z_inf_to_z_host_mean(mass_ratio)
        sig = self.z_inf_to_z_host_std(mass_ratio)
        bounds = [0.0, 15.0]
        z = float(truncnorm.rvs((bounds[0] - mu) / sig, (bounds[1] - mu) / sig,
                                 loc=mu, scale=sig))
        return self._z_lens + z

    @staticmethod
    def _mass_ratio_in_bounds(massRatio):
        """
        Make sure the ratio m_infall / m_host falls in the calibrated range 10^-5 - 10^-0.5
        :param massRatio:
        :return: the ratio m_infall / m_host
        """
        massRatio = max(10 ** -5.0, massRatio)
        massRatio = min(10 ** -0.5, massRatio)
        return massRatio

    @staticmethod
    def z_inf_to_z_host_mean(massRatio):
        """
        Return the mean infall redshift relative to the host redshift for subhalos with a mass ratio of
        massRatio = Msub / Mhost.
        :param massRatio: mass ratio of the subhao relative to the host, i.e. Msub / Mhost
        :return: the mean infall redshift relative to the host redshift
        """
        a = 3.35549949
        b = 3.20546995
        c = -2.91075587
        return a / (1 + b * (-numpy.log10(massRatio)) ** c)

    @staticmethod
    def z_inf_to_z_host_std(massRatio):
        """
        Return the standard deviation of infall redshift for subhalos with a mass ratio of
        massRatio = Msub / Mhost.
        :param massRatio: mass ratio of the subhao relative to the host, i.e. Msub / Mhost
        :return: the standard deviation of infall redshift
        """
        a = 1.97880156
        b = 4.17390702
        c = -2.14428416
        return a / (1 + b * (-numpy.log10(massRatio)) ** c)

class InfallDistributionDirect(object):
    name = 'direct'
    """Accretion redshift pdf meant for directly-infalling subhalos"""

    def __init__(self, z_lens, log_m_host):
        """

        :param z_lens: main deflector redshift
        :param log_m_host: log10 host halo mass in solar masses
        """
        self._m_host = 10 ** log_m_host
        self._z_lens = z_lens

    def __call__(self, m_sub):
        """
        Return the infall redshift for a subhalo with infall mass m_sub
        :param m_sub: infall mass in solar masses
        :return: infall redshift
        """
        mass_ratio = self._mass_ratio_in_bounds(m_sub / self._m_host)
        mu = self.z_inf_to_z_host_mean(mass_ratio)
        sig = self.z_inf_to_z_host_std(mass_ratio)
        bounds = [0.0, 15.0]
        z = float(truncnorm.rvs((bounds[0] - mu) / sig, (bounds[1] - mu) / sig,
                                 loc=mu, scale=sig))
        return self._z_lens + z

    @staticmethod
    def _mass_ratio_in_bounds(massRatio):
        """
        Make sure the ratio m_infall / m_host falls in the calibrated range 10^-5 - 10^-0.5
        :param massRatio:
        :return: the ratio m_infall / m_host
        """
        massRatio = max(10 ** -5.0, massRatio)
        massRatio = min(10 ** -0.5, massRatio)
        return massRatio

    @staticmethod
    def z_inf_to_z_host_mean(massRatio):
        """
        Return the mean infall redshift relative to the host redshift for subhalos with a mass ratio of
        massRatio = Msub / Mhost.
        :param massRatio: mass ratio of the subhao relative to the host, i.e. Msub / Mhost
        :return: the mean infall redshift relative to the host redshift
        """
        a = 3.3064042
        b = 4.528511
        c = -3.84346439
        return a / (1 + b * (-numpy.log10(massRatio)) ** c)

    @staticmethod
    def z_inf_to_z_host_std(massRatio):
        """
        Return the standard deviation of infall redshift for subhalos with a mass ratio of
        massRatio = Msub / Mhost.
        :param massRatio: mass ratio of the subhao relative to the host, i.e. Msub / Mhost
        :return: the standard deviation of infall redshift
        """
        a = 1.94929171
        b = 4.91504583
        c = -2.4187402
        return a / (1 + b * (-numpy.log10(massRatio)) ** c)
#
# pdf = InfallDistributionHybrid(0.5, 13)
# pdf_direct = InfallDistributionDirect(0.5, 13)
#
# import numpy as np
# m_sub = np.logspace(8., 9, 10000)
# z = [pdf(m_sub_i) for m_sub_i in m_sub]
# z_direct = [pdf_direct(m_sub_i) for m_sub_i in m_sub]
# import matplotlib.pyplot as plt
#
# plt.hist(z,color='k',alpha=0.5,bins=20,range=(0.5, 10),label='hybrid')
# plt.hist(z_direct, color='r', alpha=0.5,bins=20,range=(0.5, 10),label='direct')
# plt.legend(fontsize=12)
# plt.gca().annotate('m < 10^9', xy=(0.7, 0.5),
#                    fontsize=14,xycoords='axes fraction')
# plt.show()
