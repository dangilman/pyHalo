import numpy
from scipy.optimize import minimize
from lenstronomy.Cosmo.nfw_param import NFWParam
import astropy.units as un
from colossus.lss.bias import twoHaloTerm
from scipy.integrate import quad
from pyHalo.Halos.accretion import InfallDistributionGalacticus2024, InfallDistributionHybrid


class NFWParampyHalo(NFWParam):

    """Adds methods for psuedo-NFW profiles to those implemented in lenstronomy"""

    def rho0_c_pseudoNFW(self, c, z):
        """
        computes density normalization as a function of concentration parameter for an NFW-like profile
        with (1+x^2) in the denominator instead of (1+x)^2

        :param c: concentration
        :param z: redshift
        :return: density normalization in h^2/Mpc^3 (physical)
        """
        return 2 * c ** 3 * self.rhoc_z(z) * 200 / (3 * numpy.log(1+c**2))

class LensCosmo(object):

    def __init__(self, z_lens=None, z_source=None, cosmology=None,
                 infall_redshift_model='HYBRID_INFALL', kwargs_infall_model={'log_m_host': 13.0}):
        """

        This class performs calcuations relevant for certain halo mass profiles and lensing-related quantities for a
        given lens/source redshift and cosmology
        :param z_lens: deflector redshift
        :param z_source: source redshift
        :param cosmology: and instance of the Cosmology class (see pyhalo.Cosmology.cosmology.py)
        :param infall_redshift_model: a string or class that determines subhalo infall times
        :param kwargs_infall_model: keyword arguments for infall time model
        """
        if cosmology is None:
            from pyHalo.Cosmology.cosmology import Cosmology
            cosmology = Cosmology()
        self.cosmo = cosmology
        self._arcsec = 2 * numpy.pi / 360 / 3600
        self.h = self.cosmo.h
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
        self._computed_zacc_pdf = False
        self._nfw_param = NFWParampyHalo(self.cosmo.astropy)
        self.z_lens = z_lens
        self.z_source = z_source
        if infall_redshift_model is not None:
            self.setup_infall_model(infall_redshift_model, kwargs_infall_model)
        else:
            self._infall_pdf_set = False

    def setup_infall_model(self, infall_redshift_model, kwargs_infall_model):

        self._infall_pdf_set = True
        if infall_redshift_model == 'HYBRID_INFALL':
            if 'm_host' in list(kwargs_infall_model.keys()):
                kwargs_infall_model['log_m_host'] = numpy.log10(kwargs_infall_model['m_host'])
                del kwargs_infall_model['m_host']
            if 'log_m_host' not in list(kwargs_infall_model.keys()):
                print('the HYBRID_INFALL model for subhalos requires m_host or log_m_host to be passed as'
                      'keyword arguments through kwargs_infall_model. Using a default value of log_m_host = 13.0')
                kwargs_infall_model['log_m_host'] = 13.0
            self._z_infall_model = InfallDistributionHybrid(self.z_lens, kwargs_infall_model['log_m_host'])
        elif infall_redshift_model == 'DIRECT_INFALL':
            self._z_infall_model = InfallDistributionGalacticus2024(self.z_lens)
        else:
            try:
                self._z_infall_model = infall_redshift_model(self.z_lens, **kwargs_infall_model)
            except:
                if isinstance(infall_redshift_model, str):
                    raise Exception(infall_redshift_model + ' not a valid infall redshift model.')
                else:
                    raise Exception('infall_time_model must be either a class, or a string identifying a '
                                    'particular model. Current options are HYBRID_INFALL and DIRECT_INFALL.')

    def z_accreted_from_zlens(self, m_sub):
        """
        Returns the redshift a subhalo was accreted. Note that in the current implementation this is
        independent of infall mass
        :param m_sub: subhalo mass at infall
        :param z_lens: main deflector redshift
        :return: accretion redshift
        """
        if self._infall_pdf_set:
            return self._z_infall_model(m_sub)
        else:
            raise Exception('must set the infall redshift model before calculating accretion redshift')

    def two_halo_boost(self, m200, z, rmin=0.5, rmax=10):

        """
        Computes the average contribution of the two halo term in a redshift slice adjacent
        the main deflector. Returns a rescaling factor applied to the mass function normalization

        :param m200: host halo mass
        :param z: redshift
        :param rmin: lower limit of the integral, something like the virial radius ~500 kpc
        :param rmax: Upper limit of the integral, this is computed based on redshift spacing during
        the rendering of halos
        :return: scaling factor applied to the normalization of the LOS mass function
        """

        mean_boost = 2 * quad(self.twohaloterm, rmin, rmax, args=(m200, z))[0] / (rmax - rmin)
        # factor of two for symmetry in front/behind host halo

        return 1. + mean_boost

    def twohaloterm(self, r, M, z, mdef='200c'):

        """
        Computes the boost to the background density of the Universe
        from correlated structure around a host of mass M
        :param r:
        :param M:
        :param z:
        :param mdef:
        :return:
        """

        h = self.cosmo.h
        M_h = M * h
        r_h = r * h
        rho_2h = twoHaloTerm(r_h, M_h, z, mdef=mdef) / self.cosmo._colossus_cosmo.rho_m(z)
        return rho_2h

    def nfw_physical2angle(self, m, c, z, pseudo_nfw=False):
        """
        converts the physical mass and concentration parameter of an NFW profile into the lensing quantities
        updates lenstronomy implementation with arbitrary redshift

        :param m: mass enclosed 200 rho_crit in units of M_sun (physical units, meaning no little h)
        :param c: NFW concentration parameter (r200/r_s)
        :param z: redshift at which to evaluate angles from distances
        :param pseudo_nfw: specifies whether one deals with a regualr NFW profile (False) or a psuedo-NFW profile
        with (1+x^2) in the denominator rather than (1+x)^2
        :return: Rs_angle (angle at scale radius) (in units of arcsec), alpha_Rs (observed bending angle at the scale radius
        """
        rho0, Rs, _ = self.nfwParam_physical(m, c, z, pseudo_nfw)
        return self.nfw_physical2angle_fromNFWparams(rho0, Rs, z, pseudo_nfw)

    def nfw_physical2angle_fromNFWparams(self, rho0, Rs, z, pseudo_nfw=False):
        """
        computes the angular lensing properties of an NFW profile from its physical parameters
        :param rho0: central density normalization [M_sun / Mpc^3]
        :param Rs: scale radius [Mpc]
        :param z: redshift at which to evaluate angles from distances
        :param pseudo_nfw: specifies whether one deals with a regular NFW profile (False) or a psuedo-NFW profile
        with (1+x^2) in the denominator rather than (1+x)^2
        :return: scale radius and deflection angle at the scale radius in arcsec
        """
        dd = self.cosmo.D_A_z(z)
        Rs_angle = Rs / dd / self._arcsec  # Rs in arcsec
        if pseudo_nfw:
            r2 = numpy.sqrt(2)
            alpha_Rs = rho0 * (4 * Rs ** 2 * (r2*numpy.log(1+r2) + numpy.log(1. / 2.)))
        else:
            alpha_Rs = rho0 * (4 * Rs ** 2 * (1 + numpy.log(1. / 2.)))
        sigma_crit = self.get_sigma_crit_lensing(z, self.z_source)
        return Rs_angle, alpha_Rs / sigma_crit / dd / self._arcsec

    def rN_M(self, M, z, N):
        """
        computes the radius R_N of a halo of mass M in physical mass M/h, where N is a number multiplying the critical
        density of the Universe at z

        :param M: halo mass in M_sun/h
        :param z: redshift
        :param N: number, e.g. N=200 computes r200
        :return: radius R_N in physical Mpc/h
        """
        rn_mpc_over_h = (3 * M / (4 * numpy.pi * self._nfw_param.rhoc_z(z) * N)) ** (1. / 3.)
        return rn_mpc_over_h / self.cosmo.h

    def nfwParam_physical(self, m, c, z, pseudo_nfw=False):
        """
        returns the NFW parameters in physical units
        updates lenstronomy implementation with arbitrary redshift

        :param m: physical mass in M_sun
        :param c: concentration
        :return: rho0 [Msun/Mpc^3], Rs [Mpc], r200 [Mpc]
        """
        if pseudo_nfw is None:
            raise Exception('psuedo_nfw must be specified in Halo class when accessing nfw parameters!')
        r200 = self._nfw_param.r200_M(m * self.h, z) / self.h  # physical radius r200
        if pseudo_nfw:
            rho0 = self._nfw_param.rho0_c_pseudoNFW(c, z) * self.h ** 2 # physical density in M_sun/Mpc**3
        else:
            rho0 = self._nfw_param.rho0_c(c, z) * self.h**2  # physical density in M_sun/Mpc**3
        Rs = r200/c
        return rho0, Rs, r200

    def NFW_params_physical(self, m, c, z, pseudo_nfw=False):
        """
        returns the NFW parameters in physical units

        :param M: physical mass in M_sun
        :param c: concentration
        :param z: redshift
        :param pseudo_nfw: bool; if False, uses a regular NFW profile, if True, uses an NFW profile
        with (1+x^2) in the denominator
        :return: rho0 [Msun/kpc^3], Rs [kpc], r200 [kpc]
        """
        rho0, Rs, r200 = self.nfwParam_physical(m, c, z, pseudo_nfw)
        return rho0 * 1000 ** -3, Rs * 1000, r200 * 1000

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

    def mthermal_to_halfmode(self, m_thermal):
        """
        Convert thermal relic particle mass to half-mode mass
        :param m_thermaal:
        :return:
        """
        # too lazy for algebra
        def _func(m):
            return abs(self.halfmode_to_thermal(m)-m_thermal)/0.01
        return minimize(_func, x0=10**8, method='Nelder-Mead')['x']

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
        Computes thee critial density for lensing in units of M_sun / Mpc^2
        :param z1: the lens redshit
        :param z2: the source redshift
        :return: the critial density for lensing
        """
        D_ds = self.cosmo.D_A(z1, z2)
        D_d = self.cosmo.D_A_z(z1)
        D_s = self.cosmo.D_A_z(z2)
        d_inv = D_s*D_ds**-1*D_d**-1
        # (Mpc ^2 / sec^2) * (Mpc^-3 M_sun^1 sec ^ 2) * Mpc ^-1 = M_sun / Mpc ^2
        epsilon_crit = (self.cosmo.c**2*(4*numpy.pi*self.cosmo.G)**-1)*d_inv
        return epsilon_crit

    def thetaE_from_sigma(self, z, sigma):
        """
        Calculate the Einstein radius of an SIS profile from velocity dispersion
        :param sigma: velocity dispersion in km/sec
        :return: Einstein radius in arcsec
        """
        arcsec = 206265
        c = 299792.5 # in km/sec
        D_s = self.cosmo.D_A(0.0, self.z_source)
        D_ds = self.cosmo.D_A(z, self.z_source)
        return 4 * numpy.pi * (sigma / c) ** 2 * D_ds / D_s * arcsec

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

    def halo_dynamical_time(self, m_host, z, c_host):
        """
        This routine computes the dynamical timescale for a halo of mass M defined as
        t = 0.5427 / sqrt(G*rho)
        where G is the gravitational constant and rho is the average density
        :param m_host: host mass in M_sun
        :param z: host redshift
        :param c_host: host halo concentration
        :return: the dynamical timescale in Gyr
        """

        _, _, rvir = self.NFW_params_physical(m_host, c_host, z)
        volume = (4/3)*numpy.pi*rvir**3
        rho_average = m_host / volume
        g = 4.3e-6
        return 0.5427 / numpy.sqrt(g*rho_average)

    def sidm_halo_effective_age(self, z, z_infall, lambda_t, zform=10.0):
        """
        Calculates a time since z = zform t_1 + t_2 where t_1 is the time from formation to infall, and t_2
        is the time from infall to redshift z times lambda_t
        :param z: halo redshift at the time of lensing
        :param z_infall: infall redshift
        :param lambda_t: rescales the passage of time since the halo becomes a subhalo
        :param zform: formation redshift
        :return: "age" in Gyr
        """
        if z_infall > 10:
            z_infall = 10
        time_formation_to_infall = self.cosmo.halo_age(z_infall, zform=zform)
        time_infall_to_z = self.cosmo.halo_age(z, zform=z_infall)
        return time_formation_to_infall + lambda_t * time_infall_to_z

    def sidm_collapse_timescale(self, rhos, rs, sigma_eff):
        """
        Calculate the SIDM timescale from NFW halo parameters and an effective cross section (Essig et al. 2019)
        :param rhos: NFW halo density scale
        :param rs: NFW halo scale radius
        :param sigma_eff: an "effective" cross section
        :return: timescale in Gyr
        """
        C = 0.75
        G = 4.3e-6 # kpc/M (km/sec)^2
        const1 = 2.0889e-10 # one cm^2 per gram in kpc^2 per M_sun
        const2 = 1.05e-33 # kpc km^2 / s^2 / solar mass in kpc^3 / m_sun / s^2
        denom = const1 * sigma_eff * rhos * rs * numpy.sqrt(4 * numpy.pi * G * rhos * const2)
        tc_seconds = 150 / C / denom
        tc_gyr = tc_seconds * 3.171e-17
        return tc_gyr

    @staticmethod
    def nfw_vmax(rhos, rs):
        """
        Calculate vmax for an NFW profile
        :param rhos: density normalization [solar mass / kpc^3]
        :param rs: scale radius [kpc]
        :return: vmax [km/sec]
        """
        G = 4.3e-6 # kpc / solar mass * (km/sec)^2
        vmax = 1.64 * numpy.sqrt(G * rhos * rs ** 2)
        return vmax
