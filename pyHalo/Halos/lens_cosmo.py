import numpy
from scipy.interpolate import interp1d
from scipy.special import erfc
from pyHalo.Halos.structural_parameters import HaloStructure

class LensCosmo(object):

    G = 4.3e-6 # kpc / solar mass (km/sec)^2

    def __init__(self, z_lens, z_source, cosmology):

        self.cosmo = cosmology
        self.z_lens, self.z_source = z_lens, z_source
        # critical density of the universe in M_sun Mpc^-3
        self.rhoc = self.cosmo.astropy.critical_density0.value * \
                    self.cosmo.density_to_MsunperMpc / self.cosmo.h ** 2

        # critical density for lensing in units M_sun * Mpc ^ -2
        self.epsilon_crit = self.get_epsiloncrit(z_lens, z_source)
        # critical density for lensing in units M_sun * kpc ^ -2
        self.epsilon_crit_kpc = self.epsilon_crit * (0.001) ** 2
        # critical density for lensing in units M_sun * arcsec ^ -2 at lens redshift
        self.sigmacrit = self.epsilon_crit * (0.001) ** 2 * self.cosmo.kpc_per_asec(z_lens) ** 2
        # lensing distances
        self.D_d, self.D_s, self.D_ds = self.cosmo.D_A_z(z_lens), self.cosmo.D_A_z(z_source), self.cosmo.D_A(
            z_lens, z_source)
        # hubble distance in Mpc
        self._d_hubble = self.cosmo.c * self.cosmo.Mpc * 0.001 * (self.cosmo.h * 100)

        self._kpc_per_arcsec_zlens = self.cosmo.kpc_per_asec(self.z_lens)

        self._halo_structure = HaloStructure(self)

    def sigma_crit_mass(self, z, geometry):

        area = geometry.angle_to_physical_area(0.5 * geometry.cone_opening_angle, z)
        sigma_crit_mpc = self.get_epsiloncrit(z, geometry._zsource)

        return area * sigma_crit_mpc

    @property
    def _subhalo_accretion_pdfs(self):

        if not hasattr(self, '_mlist') or not hasattr(self, '_dzvals') \
        or not hasattr(self, '_cdfs'):

            self._mlist, self._dzvals, self._cdfs = self._Msub_cdfs(self.z_lens)

        return self._mlist, self._dzvals, self._cdfs

    @property
    def colossus(self):
        return self.lens_cosmo.colossus

    ######################################################
    """ACCESS ROUTINES IN STRUCTURAL PARAMETERS CLASS"""
    ######################################################

    def pericenter_given_r3d(self, r3d):
        return self._halo_structure._pericenter_given_r3d(r3d)

    def truncation_roche(self, args):
        return self._halo_structure.truncation_roche(*args)

    def truncation_mean_density(self, args):
        return self._halo_structure.truncation_mean_density_NFW_host(*args)

    def truncation_mean_density_isothermal_host(self, args):
        return self._halo_structure.truncation_mean_density_isothermal_host(*args)

    def LOS_truncation(self, M, z, N=50):
        return self._halo_structure.LOS_truncation(M, z, N)

    def NFW_concentration(self, M, z, model='diemer19', mdef='200c', logmhm=0,
                          scatter=True, c_scale=None, c_power=None, scatter_amplitude=0.13):

        return self._halo_structure._NFW_concentration(M, z, model, mdef, logmhm,
                          scatter, c_scale, c_power, scatter_amplitude)

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

        return 10**9 * (m_thermal / 2.3) ** (-3.33)

    def halfmode_to_thermal(self, m_half_mode):

        """
        Converts a half mode mass in units of solar masses (no little h) to the mass of
        the corresponding thermal relic particle in keV
        :param m: half mode mass in solar masses
        :return: thermal relic particle mass in keV
        """

        return 2.3 * (m_half_mode / 10**9) ** (-0.3)

    ##################################################################################
    """LENSING ROUTINES"""
    ##################################################################################

    def get_epsiloncrit(self,z1,z2):

        D_ds = self.cosmo.D_A(z1, z2)
        D_d = self.cosmo.D_A_z(z1)
        D_s = self.cosmo.D_A_z(z2)

        epsilon_crit = (self.cosmo.c**2*(4*numpy.pi*self.cosmo.G)**-1)*(D_s*D_ds**-1*D_d**-1)

        return epsilon_crit

    def get_epsiloncrit_kpc(self, z1, z2):

        return self.get_epsiloncrit(z1, z2) * 0.001 ** 2

    def get_sigmacrit(self, z):

        return self.get_epsiloncrit(z,self.z_source)*(0.001)**2*self.cosmo.kpc_per_asec(z)**2

    def get_sigmacrit_z1z2(self,zlens,zsrc):

        return self.get_epsiloncrit(zlens,zsrc)*(0.001)**2*self.cosmo.kpc_per_asec(zlens)**2

    def nfw_physical2angle(self, M, c, z):
        """
        converts the physical mass and concentration parameter of an NFW profile into the lensing quantities
        :param M: mass enclosed 200 \rho_crit
        :param c: NFW concentration parameter (r200/r_s)
        :return: theta_Rs (observed bending angle at the scale radius, Rs_angle (angle at scale radius) (in units of arcsec)
        """
        D_d = self.cosmo.D_A_z(z)
        rho0, Rs, r200 = self._nfwParam_physical_Mpc(M, c, z)

        Rs_angle = Rs / D_d / self.cosmo.arcsec  # Rs in arcsec
        theta_Rs = rho0 * (4 * Rs ** 2 * (1 + numpy.log(1. / 2.)))
        eps_crit = self.get_epsiloncrit(z, self.z_source)
        return Rs_angle, theta_Rs / eps_crit / D_d / self.cosmo.arcsec

    def rho0_c_NFW(self, c):
        """
        computes density normalization as a function of concentration parameter
        :return: density normalization in h^2/Mpc^3 (comoving)
        """

        return 200. / 3 * self.rhoc * c ** 3 / (numpy.log(1 + c) - c / (1 + c))

    def rN_M_nfw_comoving(self, M, N):
        """
        computes the radius R_N of a halo of mass M in comoving distances
        :param M: halo mass in M_sun
        :type M: float or numpy array
        :return: radius R_200 in comoving Mpc/h
        """

        return (3 * M / (4 * numpy.pi * self.rhoc * N)) ** (1. / 3.)

    def _nfwParam_physical_Mpc(self, M, c, z):

        h = self.cosmo.h
        a_z = (1+z) ** -1
        r200 = self.rN_M_nfw_comoving(M * h, 200) / h * a_z  # physical radius r200
        rho0 = self.rho0_c_NFW(c) * h ** 2 / a_z ** 3  # physical density in M_sun/Mpc**3
        Rs = r200 / c
        return rho0, Rs, r200

    def NFW_params_physical(self, M, c, z):
        """
        :param M: physical M200
        :param c: concentration
        :param z: halo redshift
        :return: physical NFW parameters in kpc units
        """
        rho0, Rs, r200 = self._nfwParam_physical_Mpc(M, c, z)

        return rho0 * 1000 ** -3, Rs * 1000, r200 * 1000

    def NFW_params_physical_fromM(self, M, z, mc_kwargs={}):

        c = self.NFW_concentration(M, z, scatter=False, **mc_kwargs)
        return self.NFW_params_physical(M, c, z)

    def vdis_to_Rein(self, zd, zsrc, vdis):

        return 4 * numpy.pi * (vdis * (0.001 * self.cosmo.c * self.cosmo.Mpc) ** -1) ** 2 * \
               self.cosmo.D_A(zd, zsrc) * self.cosmo.D_A_z(zsrc) ** -1 * self.cosmo.arcsec ** -1

    @property
    def point_mass_factor(self):

        factor = 4 * self.cosmo.G * self.cosmo.c ** -2 * \
                 self.D_ds * (self.D_d * self.D_s) ** -1
        return factor ** 0.5 / self.cosmo.arcsec

    ##################################################################################
    """ACCRETION REDSHIFT PDF FROM GALACTICUS"""
    ##################################################################################
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

    # def SIDMrho(self, cross_section, sidm_func, halo_mass, halo_redshift, cscatter = True):
    #
    #     cmean = self.NFW_concentration(halo_mass, halo_redshift, scatter=False)
    #     if cscatter:
    #         c = self.NFW_concentration(halo_mass, halo_redshift)
    #     else:
    #         c = cmean
    #
    #     rho, _, _ = self.NFW_params_physical(halo_mass, c, halo_redshift)
    #
    #     zeta = self.lens_cosmo.cosmo.halo_age(halo_redshift) * cross_section
    #
    #     rho_sidm = 10**sidm_func(halo_mass, halo_redshift, zeta, cmean, c)
    #
    #     return rho_sidm, rho / rho_sidm

    # def NFWv200_fromM(self, M, z, mc_scatter=False):
    #
    #     _, _, r200 = self.NFW_params_physical_fromM(M, z, mc_scatter=mc_scatter)
    #
    #     return numpy.sqrt(self.G * M / r200)

    # def NFWvmax_fromM(self, M, z, mc_scatter=False):
    #
    #     c = self.NFW_concentration(M, z, scatter=mc_scatter)
    #
    #     _, _, r200 = self.NFW_params_physical(M, c, z)
    #
    #     vmax = numpy.sqrt(self.G * M / r200)
    #
    #     return vmax * (0.216 * (numpy.log(1 + c) - c * (1+c) ** -1) * c ** -1) ** 0.5
