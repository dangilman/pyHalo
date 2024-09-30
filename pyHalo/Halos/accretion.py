import numpy
from scipy.interpolate import interp1d
from scipy.stats import truncnorm


class InfallDistributionGalacticus2024(object):
    """ACCRETION REDSHIFT PDF FROM GALACTICUS USING THE VERSION OF GALACTICUS AS OF FEB 2024 AND SELECTING ON
    SUBHALOS WITH A BOUND MASS > 10^6"""
    name = 'direct'
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

        :param massRatio:
        :return:
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

# class InfallDistributionGalacticus2024(object):
#     """ACCRETION REDSHIFT PDF FROM GALACTICUS USING THE VERSION OF GALACTICUS AS OF FEB 2024"""
#
#     def __init__(self, z_lens):
#         self.z_lens = z_lens
#         self._counts = numpy.array([ 74, 111, 138, 225, 281, 394, 396, 492, 603, 665, 626, 738, 714,
#         725, 744, 712, 679, 600, 556, 524, 478, 442, 347, 322, 283, 198,
#         189, 148, 137,  98,  77,  44,  32,  32,  26,  18,  15,   6,   5,
#         4,   0,   2,   0,   0])
#         self._z_infall = numpy.array([ 0.25,  0.75,  1.25,  1.75,  2.25,  2.75,  3.25,  3.75,  4.25,
#         4.75,  5.25,  5.75,  6.25,  6.75,  7.25,  7.75,  8.25,  8.75,
#         9.25,  9.75, 10.25, 10.75, 11.25, 11.75, 12.25, 12.75, 13.25,
#         13.75, 14.25, 14.75, 15.25, 15.75, 16.25, 16.75, 17.25, 17.75,
#         18.25, 18.75, 19.25, 19.75, 20.25, 20.75, 21.25, 21.75])
#         cdf = numpy.cumsum(self._counts)
#         self._cdf = cdf / numpy.max(cdf)
#         self._cdf_min = numpy.min(self._cdf)
#         self._cdf_max = numpy.max(self._cdf)
#         self._interp = interp1d(self._cdf, self._z_infall)
#
#     def z_accreted_from_zlens(self, z_lens):
#         u = numpy.random.uniform(self._cdf_min, self._cdf_max)
#         z_infall = z_lens + self._interp(u)
#         return z_infall
#
# class InfallDistributionGalacticus2020(object):
#     """ACCRETION REDSHIFT PDF FROM GALACTICUS USING THE VERSION OF GALACTICUS PUBLISHED IN 2020 WITH
#     WARM DARK MATTER CHILLS OUT"""
#     def __init__(self, z_lens):
#         self.z_lens = z_lens
#
#     @property
#     def _subhalo_accretion_pdfs(self):
#
#         if self._computed_zacc_pdf is False:
#             self._computed_zacc_pdf = True
#             self._mlist, self._dzvals, self._cdfs = self._Msub_cdfs(self.z_lens)
#
#         return self._mlist, self._dzvals, self._cdfs
#
#     def z_accreted_from_zlens(self, msub, zlens):
#
#         mlist, dzvals, cdfs = self._subhalo_accretion_pdfs
#
#         idx = self._mass_index(msub, mlist)
#
#         z_accreted = zlens + self._sample_cdf_single(cdfs[idx])
#
#         return z_accreted
#
#     def _cdf_numerical(self, m, z_lens, delta_z_values):
#
#         c_d_f = []
#
#         prob = 0
#         for zi in delta_z_values:
#             prob += self._P_fit_diff_M_sub(z_lens + zi, z_lens, m)
#             c_d_f.append(prob)
#         return numpy.array(c_d_f) / c_d_f[-1]
#
#     def _Msub_cdfs(self, z_lens):
#
#         M_sub_exp = numpy.arange(6.0, 10.2, 0.2)
#         M_sub_list = 10 ** M_sub_exp
#         delta_z = numpy.linspace(0., 6, 8000)
#         funcs = []
#
#         for mi in M_sub_list:
#             # cdfi = P_fit_diff_M_sub_cumulative(z_lens+delta_z, z_lens, mi)
#             cdfi = self._cdf_numerical(mi, z_lens, delta_z)
#
#             funcs.append(interp1d(cdfi, delta_z))
#
#         return M_sub_list, delta_z, funcs
#
#     def z_decay_mass_dependence(self, M_sub):
#         # Mass dependence of z_decay.
#         a = 3.21509397
#         b = 1.04659814e-03
#
#         return a - b * numpy.log(M_sub / 1.0e6) ** 3
#
#     def z_decay_exp_mass_dependence(self, M_sub):
#         # Mass dependence of z_decay_exp.
#
#         a = 0.30335749
#         b = 3.2777e-4
#
#         return a - b * numpy.log(M_sub / 1.0e6) ** 3
#
#     def _P_fit_diff_M_sub(self, z, z_lens, M_sub):
#         # Given the redhsift of the lens, z_lens, and the subhalo mass, M_sub, return the
#         # posibility that the subhlao has an accretion redhisft of z.
#
#         z_decay = self.z_decay_mass_dependence(M_sub)
#         z_decay_exp = self.z_decay_exp_mass_dependence(M_sub)
#
#         normalization = 2.0 / numpy.sqrt(2.0 * numpy.pi) / z_decay \
#                         / numpy.exp(0.5 * z_decay ** 2 * z_decay_exp ** 2) \
#                         / erfc(z_decay * z_decay_exp / numpy.sqrt(2.0))
#         return normalization * numpy.exp(-0.5 * ((z - z_lens) / z_decay) ** 2) \
#                * numpy.exp(-z_decay_exp * (z - z_lens))
#
#     def _sample_cdf_single(self, cdf_interp):
#
#         u = numpy.random.uniform(0, 1)
#
#         try:
#             output = float(cdf_interp(u))
#             if numpy.isnan(output):
#                 output = 0
#
#         except:
#             output = 0
#
#         return output
#
#     def _mass_index(self, subhalo_mass, mass_array):
#
#         idx = numpy.argmin(numpy.absolute(subhalo_mass - mass_array))
#         return idx

