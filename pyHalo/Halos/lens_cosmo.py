import numpy
from scipy.interpolate import interp1d
from scipy.special import erfc
from pyHalo.Halos.concentration import Concentration
import astropy.units as un

class LensCosmo(object):

    def __init__(self, z_lens=None, z_source=None, cosmology=None):

        if cosmology is None:
            from pyHalo.Cosmology.cosmology import Cosmology
            cosmology = Cosmology()

        self.cosmo = cosmology
        self.z_lens, self.z_source = z_lens, z_source

        # critical density of the universe in M_sun h^2 Mpc^-3
        rhoc = un.Quantity(self.cosmo.astropy.critical_density(0), unit=un.Msun / un.Mpc ** 3).value
        self.rhoc = rhoc / self.cosmo.h ** 2

        if z_lens is not None and z_source is not None:
            # critical density for lensing in units M_sun * Mpc ^ -2
            self.sigma_crit_lensing = self.get_sigma_crit_lensing(z_lens, z_source)
            # critical density for lensing in units M_sun * kpc ^ -2
            self.sigma_crit_lens_kpc = self.sigma_crit_lensing * (0.001) ** 2
            # critical density for lensing in units M_sun * arcsec ^ -2 at lens redshift
            self.sigmacrit = self.sigma_crit_lensing * (0.001) ** 2 * self.cosmo.kpc_proper_per_asec(z_lens) ** 2
            # lensing distances
            self.D_d, self.D_s, self.D_ds = self.cosmo.D_A_z(z_lens), self.cosmo.D_A_z(z_source), self.cosmo.D_A(
                z_lens, z_source)

        self._concentration = Concentration(self)

        self._computed_zacc_pdf = False

    def sigma_crit_mass(self, z, area):

        """
        :param z: redshift
        :param area: physical area in Mpc^2
        :return: the 'critical density mass' sigma_crit * A in units M_sun
        """

        sigma_crit_mpc = self.get_sigma_crit_lensing(z, self.z_source)

        return area * sigma_crit_mpc

    @property
    def colossus(self):
        return self.cosmo.colossus

    ######################################################
    """ACCESS ROUTINES IN STRUCTURAL PARAMETERS CLASS"""
    ######################################################

    @staticmethod
    def truncation_roche(M, r3d, k, nu):

        """
        :param M: m200
        :param r3d: 3d radial position in the halo (kpc)
        :return: truncation radius in Kpc (physical)
        """

        m_units_7 = M / 10 ** 7
        radius_units_50 = r3d / 50
        rtrunc_kpc = k * m_units_7 ** (1. / 3) * radius_units_50 ** nu

        return numpy.round(rtrunc_kpc, 3)

    def LOS_truncation_rN(self, M, z, N):
        """
        Truncate LOS halos at r50
        :param M:
        :param c:
        :param z:
        :param N:
        :return:
        """
        a_z = self.cosmo.scale_factor(z)
        h = self.cosmo.h
        r50_physical_Mpc = self.rN_M_nfw_comoving(M * h, N, z) * a_z / h
        rN_physical_kpc = r50_physical_Mpc * 1000

        return rN_physical_kpc

    def NFW_concentration(self, M, z, model='diemer19', mdef='200c', logmhm=None,
                          scatter=True, scatter_amplitude=0.2, kwargs_suppresion=None, suppression_model=None):

        """
        Returns the concentration of an NFW halo (see method in the class Concentration)
        :param M: mass in units M_solar (no little h)
        :param z: redshift
        :param model: the model for the concentration-mass relation
        if type dict, will assume a custom MC relation parameterized by c0, beta, zeta (see _NFW_concentration_custom)
        if string, will use the corresponding concentration model in colossus (see http://www.benediktdiemer.com/code/colossus/)

        :param mdef: mass defintion for use in colossus modules. Default is '200c', or 200 times rho_crit(z)
        :param logmhm: log10 of the half-mode mass in units M_sun, specific to warm dark matter models.
        This parameter defaults to 0. in the code, which leaves observables unaffected for the mass scales of interest ~10^7
        :param scatter: bool; if True will induce lognormal scatter in the MC relation with amplitude
        scatter_amplitude in dex
        :param kwargs_suppresion: keyword arguments for the suppression function
        :param suppression_model: the type of suppression, either 'polynomial' or 'hyperbolic'
        :param scatter_amplitude: the amplitude of the scatter in the mass-concentration relation in dex
        :return: the concentration of the NFW halo

        """

        return self._concentration.nfw_concentration(M, z, model, mdef, logmhm,
                                                     scatter,scatter_amplitude, kwargs_suppresion, suppression_model)

    ###############################################################
    """ROUTINES BASED ON CERTAIN COSMOLOGICAL MODELS (E.G. WDM)"""
    ###############################################################
    def mthermal_to_halfmode(self, m_thermal):

        """
        Converts a (fully thermalized) thermal relic particle of mass m [keV] to
        the half-mode mass scale in solar masses (no little h)
        :param m: thermal relic particle mass in keV
        :return: half mode mass in solar masses
        """
        # scaling of 3.3 keV from Viel et al
        omega_matter = self.cosmo.astropy.Om0
        return 10**9 * ((omega_matter/0.25)**-0.4 * (self.cosmo.h/0.7)**-0.8 *m_thermal / 2.32) ** (-3.33)

    def halfmode_to_thermal(self, m_half_mode):

        """
        Converts a half mode mass in units of solar masses (no little h) to the mass of
        the corresponding thermal relic particle in keV
        :param m: half mode mass in solar masses
        :return: thermal relic particle mass in keV
        """

        omega_matter = self.cosmo.astropy.Om0
        return 2.32 * (omega_matter / 0.25)**0.4 * (self.cosmo.h/0.7)**0.8 * \
               (m_half_mode / 10 ** 9) ** (-0.3)

    def mhm_to_fsl(self, m_hm):
        """
        Converts half mode mass to free streaming length in Mpc
        See Equations 5-8 in https://arxiv.org/pdf/1112.0330.pdf
        :param m_hm: half-mode mass in units M_sun (no little h)
        :return: free streaming length in units Mpc
        """

        rhoc = self.rhoc * self.cosmo.h ** 2

        l_hm = 2 * (3 * m_hm / (4 * numpy.pi * rhoc)) ** (1. / 3)

        l_fs = l_hm / 13.93

        return l_fs

    ##################################################################################
    """ROUTINES RELATED TO LENSING STUFF"""
    ##################################################################################

    def get_sigma_crit_lensing(self, z1, z2):

        """

        :param z1: redshift lens
        :param z2: redshift source
        :return: critical density for lensing in units of M_sun / Mpc ^ 2
        """

        D_ds = self.cosmo.D_A(z1, z2)
        D_d = self.cosmo.D_A_z(z1)
        D_s = self.cosmo.D_A_z(z2)

        d_inv = D_s*D_ds**-1*D_d**-1

        # (Mpc ^2 / sec^2) * (Mpc^-3 M_sun^1 sec ^ 2) * Mpc ^-1 = M_sun / Mpc ^2
        epsilon_crit = (self.cosmo.c**2*(4*numpy.pi*self.cosmo.G)**-1)*d_inv

        return epsilon_crit

    ##################################################################################
    """Routines relevant for NFW profiles"""
    ##################################################################################
    def NFW_params_physical(self, M, c, z):
        """

        :param M: physical M200
        :param c: concentration
        :param z: halo redshift
        :return: physical NFW parameters in kpc units
        """

        rho0, Rs, r200 = self.nfwParam_physical_Mpc(M, c, z)

        return rho0 * 1000 ** -3, Rs * 1000, r200 * 1000

    def nfw_physical2angle_fromNFWparams(self, rhos, rs, z):

        """
        computes the deflection angle properties of an NFW halo from the density normalization mass and scale radius
        :param rhos: central density normalization in M_sun / Mpc^3
        :param rs: scale radius in Mpc
        :param z: redshift
        :return: theta_Rs (observed bending angle at the scale radius, Rs_angle (angle at scale radius) (in units of arcsec)
        """

        D_d = self.cosmo.D_A_z(z)
        Rs_angle = rs / D_d / self.cosmo.arcsec  # Rs in arcsec
        theta_Rs = rhos * (4 * rs ** 2 * (1 + numpy.log(1. / 2.)))
        eps_crit = self.get_sigma_crit_lensing(z, self.z_source)

        return Rs_angle, theta_Rs / eps_crit / D_d / self.cosmo.arcsec

    def nfw_physical2angle(self, M, c, z):
        """
        converts the physical mass and concentration parameter of an NFW profile into the lensing quantities
        :param M: mass enclosed 200 \rho_crit
        :param c: NFW concentration parameter (r200/r_s)
        :return: theta_Rs (observed bending angle at the scale radius, Rs_angle (angle at scale radius) (in units of arcsec)
        """

        rhos, rs, _ = self.nfwParam_physical_Mpc(M, c, z)

        return self.nfw_physical2angle_fromNFWparams(rhos, rs, z)

    def nfw_physical2angle_fromM(self, M, z, **mc_kwargs):
        """
        converts the physical mass and concentration parameter of an NFW profile into the lensing quantities
        (with no scatter in MC relation)
        :param M: mass enclosed 200 \rho_crit(z)
        :param z: redshift
        :param mc_kwargs: keyword arguments for NFW_concentration
        :return: theta_Rs (observed bending angle at the scale radius, Rs_angle (angle at scale radius) (in units of arcsec)
        """

        c = self.NFW_concentration(M, z, scatter=False, **mc_kwargs)

        return self.nfw_physical2angle(M, c, z)

    def rho0_c_NFW(self, c, z_eval_rho=0.):
        """
        computes density normalization as a function of concentration parameter
        :return: density normalization in h^2/Mpc^3 (comoving)
        """

        rho_crit = self.cosmo.rho_crit(z_eval_rho) / self.cosmo.h ** 2
        return 200. / 3 * rho_crit * c ** 3 / (numpy.log(1 + c) - c / (1 + c))

    def rN_M_nfw_comoving(self, M, N, z):
        """
        computes the radius R_N of a halo of mass M in comoving distances
        :param M: halo mass in M_sun/h
        :type M: float or numpy array
        :return: radius R_200 in comoving Mpc/h
        """

        rho_crit = self.cosmo.rho_crit(z) / self.cosmo.h ** 2

        return (3 * M / (4 * numpy.pi * rho_crit * N)) ** (1. / 3.)

    def nfwParam_physical_Mpc(self, M, c, z):

        """

        :param M: halo mass in units M_sun (no little h)
        :param c: concentration parameter
        :param z: redshift
        :return: physical rho_s, rs for the NFW profile in physical units M_sun (no little h), Mpc

        Mass definition critical density of Universe with respect to critical density at redshift z
        Also specified in colossus as 200c
        """

        h = self.cosmo.h
        r200 = self.rN_M_nfw_comoving(M * h, 200., z) / h  # physical radius r200
        rhos = self.rho0_c_NFW(c, z) * h ** 2  # physical density in M_sun/Mpc**3
        rs = r200 / c
        return rhos, rs, r200

    ##################################################################################
    """Routines relevant for other lensing by other mass profiles"""
    ##################################################################################

    def point_mass_factor_z(self, z):

        """
        Returns the cosmology-dependent factor to evaluate the Einstein radius of a point mass of mass M:

        :param z: redshift
        :return: The factor that when multiplied by sqrt(mass) gives the Einstein radius of a point mass

        R_ein = sqrt(M) * point_mass_factor_z(z)

        """
        factor = 4 * self.cosmo.G * self.cosmo.c ** -2

        dds = self.cosmo.D_A(z, self.z_source)
        dd = self.cosmo.D_A_z(z)
        ds = self.D_s

        factor *= dds / dd / ds

        return factor ** 0.5 / self.cosmo.arcsec

    ##################################################################################
    """ACCRETION REDSHIFT PDF FROM GALACTICUS"""
    ##################################################################################

    @property
    def _subhalo_accretion_pdfs(self):

        if self._computed_zacc_pdf is False:

            self._computed_zacc_pdf = True
            self._mlist, self._dzvals, self._cdfs = self._Msub_cdfs(self.z_lens)

        return self._mlist, self._dzvals, self._cdfs

    def z_accreted_from_zlens(self, msub, zlens):

        mlist, dzvals, cdfs = self._subhalo_accretion_pdfs

        idx = self._mass_index(msub, mlist)

        z_accreted = zlens + self._sample_cdf_single(cdfs[idx])

        return z_accreted

    def _cdf_numerical(self, m, z_lens, delta_z_values):

        c_d_f = []

        prob = 0
        for zi in delta_z_values:
            prob += self._P_fit_diff_M_sub(z_lens + zi, z_lens, m)
            c_d_f.append(prob)
        return numpy.array(c_d_f) / c_d_f[-1]

    def _Msub_cdfs(self, z_lens):

        M_sub_exp = numpy.arange(6.0, 10.2, 0.2)
        M_sub_list = 10 ** M_sub_exp
        delta_z = numpy.linspace(0., 6, 8000)
        funcs = []

        for mi in M_sub_list:
            # cdfi = P_fit_diff_M_sub_cumulative(z_lens+delta_z, z_lens, mi)
            cdfi = self._cdf_numerical(mi, z_lens, delta_z)

            funcs.append(interp1d(cdfi, delta_z))

        return M_sub_list, delta_z, funcs

    def z_decay_mass_dependence(self, M_sub):
        # Mass dependence of z_decay.
        a = 3.21509397
        b = 1.04659814e-03

        return a - b * numpy.log(M_sub / 1.0e6) ** 3

    def z_decay_exp_mass_dependence(self, M_sub):
        # Mass dependence of z_decay_exp.

        a = 0.30335749
        b = 3.2777e-4

        return a - b * numpy.log(M_sub / 1.0e6) ** 3

    def _P_fit_diff_M_sub(self, z, z_lens, M_sub):
        # Given the redhsift of the lens, z_lens, and the subhalo mass, M_sub, return the
        # posibility that the subhlao has an accretion redhisft of z.

        z_decay = self.z_decay_mass_dependence(M_sub)
        z_decay_exp = self.z_decay_exp_mass_dependence(M_sub)

        normalization = 2.0 / numpy.sqrt(2.0 * numpy.pi) / z_decay \
                        / numpy.exp(0.5 * z_decay ** 2 * z_decay_exp ** 2) \
                        / erfc(z_decay * z_decay_exp / numpy.sqrt(2.0))
        return normalization * numpy.exp(-0.5 * ((z - z_lens) / z_decay) ** 2) \
               * numpy.exp(-z_decay_exp * (z - z_lens))

    def _sample_cdf_single(self, cdf_interp):

        u = numpy.random.uniform(0, 1)

        try:
            output = float(cdf_interp(u))
            if numpy.isnan(output):
                output = 0

        except:
            output = 0

        return output

    def _mass_index(self, subhalo_mass, mass_array):

        idx = numpy.argmin(numpy.absolute(subhalo_mass - mass_array))
        return idx

