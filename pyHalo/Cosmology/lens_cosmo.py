from pyHalo.Cosmology.cosmology import Cosmology
from colossus.halo.concentration import concentration
import numpy
from scipy.integrate import quad

class LensCosmo(object):

    interp_redshifts = numpy.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    interp_values = [[0.79871744, 0.1385], [0.85326291, 0.2048],
                     [0.87740329, 0.2479], [0.87212949, 0.31635],
                     [0.90235063, 0.36065], [0.85694561, 0.4112],
                     [0.8768506, 0.4522]]
    mass_norm = 13.0
    norm_norm = 0.01

    # values calibrated from Galacticus runs inside rmax = 20kpc

    def __init__(self,z_lens,z_source):

        self.cosmo = Cosmology()
        self.z_lens, self.z_source = z_lens, z_source
        # critical density of the universe in M_sun Mpc^-3
        self.rhoc = self.cosmo.astropy.critical_density0.value * self.cosmo.density_to_MsunperMpc

        if z_lens != 0:
            # critical density for lensing in units M_sun * Mpc ^ -2
            self.epsilon_crit = self.get_epsiloncrit(z_lens, z_source)
            # critical density for lensing in units M_sun * arcsec ^ -2 at lens redshift
            self.sigmacrit = self.epsilon_crit * (0.001) ** 2 * self.cosmo.kpc_per_asec(z_lens) ** 2
            # lensing distances
            self.D_d, self.D_s, self.D_ds = self.cosmo.D_A(0, z_lens), self.cosmo.D_A(0, z_source), self.cosmo.D_A(z_lens, z_source)
            # hubble distance in Mpc
            self._d_hubble = self.cosmo.c * self.cosmo.Mpc * 0.001 * (self.cosmo.h * 100)

            self._kpc_per_asec_zlens = self.cosmo.kpc_per_asec(self.z_lens)

    def _eval_halo_interp(self, halo_z, log_halo_mass):

        logm_norm = log_halo_mass * self.mass_norm ** -1

        if halo_z < self.interp_redshifts[0]:
            raise Exception('main lens halo at a higher redshift than interpolated values cover.')
        elif halo_z > self.interp_redshifts[-1]:
            raise Exception('main lens halo at a higher redshift than interpolated values cover.')
        else:
            argmins = numpy.argsort(numpy.absolute(halo_z - self.interp_redshifts))[0:2]
            expon1 = self.interp_values[argmins[0]][0] * logm_norm + self.interp_values[argmins[0]][1]
            expon2 = self.interp_values[argmins[1]][0] * logm_norm + self.interp_values[argmins[1]][1]
            w1 = 1 - numpy.absolute(halo_z - self.interp_redshifts[argmins[0]]) * 0.1 ** -1
            w2 = 1 - w1

            norm1 = 10 ** expon1
            norm2 = 10 ** expon2

            return self.norm_norm * (w1 * norm1 + w2 * norm2)

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

    def get_epsiloncrit(self,z1,z2):

        D_ds = self.cosmo.D_A(z1, z2)
        D_d = self.cosmo.D_A(0, z1)
        D_s = self.cosmo.D_A(0, z2)

        epsilon_crit = (self.cosmo.c**2*(4*numpy.pi*self.cosmo.G)**-1)*(D_s*D_ds**-1*D_d**-1)

        return epsilon_crit

    def get_sigmacrit(self, z):

        return self.get_epsiloncrit(z,self.z_source)*(0.001)**2*self.cosmo.kpc_per_asec(z)**2

    def get_sigmacrit_z1z2(self,zlens,zsrc):

        return self.get_epsiloncrit(zlens,zsrc)*(0.001)**2*self.cosmo.kpc_per_asec(zlens)**2

    def vdis_to_Rein(self,zd,zsrc,vdis):

        return 4 * numpy.pi * (vdis * (0.001 * self.cosmo.c * self.cosmo.Mpc) ** -1) ** 2 * \
               self.cosmo.D_A(zd, zsrc) * self.cosmo.D_A(0,zsrc) ** -1 * self.cosmo.arcsec ** -1

    def vdis_to_Reinkpc(self,zd,zsrc,vdis):

        return self.cosmo.kpc_per_asec(zd)*self.cosmovdis_to_Rein(zd,zsrc,vdis)

    def beta(self,z,zmain,zsrc):

        D_12 = self.cosmo.D_A(zmain, z)
        D_os = self.cosmo.D_A(0, zsrc)
        D_1s = self.cosmo.D_A(zmain, zsrc)
        D_o2 = self.cosmo.D_A(0, z)

        return D_12 * D_os * (D_o2 * D_1s) ** -1

    def subhalo_mfunc_spatial_scaing(self, R_ein_arcsec, zlens):

        """
               This function returns a scaling factor that accounts for an angular position inside
               a halo as a function of redshift and Einstein radius. I assume M_halo = 10^13.
               The scaling factor returned is normalized with respect to the value at redshift
               0.5 and Einstein radius of 1 arcsec.
               :param R_ein_arcsec: Einstein radius in arcsec
               :param zlens: lens redshift
               @ z = 0.5  and  R_ein  = 1 arcsec:
               rs = 80kpc
               kappa_sub(R_ein/r50) = 4.554 (units arbitrary since it is a scaling relation)
               """

        # reference values
        kappa_nfw_reference = 4.554
        a_z_reference = (1 + 0.5) ** -1
        rs_reference = 80  # kpc

        # rescale to correct redshift
        a_z = (1 + zlens) ** -1
        a_z_rescale = a_z * a_z_reference ** -1
        Rs = rs_reference * a_z_rescale

        # Einstein radius in kpc
        R_ein_kpc = R_ein_arcsec * self.cosmo.kpc_per_asec(zlens)
        x = R_ein_kpc * Rs ** -1

        # new kappa value
        kappa_nfw = 2 * (1 - numpy.arctanh(numpy.sqrt(1 - x ** 2)) *
                         numpy.sqrt(1 - x ** 2) ** -1) * (x ** 2 - 1) ** -1

        # compute the scaling factor
        kappa_scaling = kappa_nfw * kappa_nfw_reference ** -1
        scaling = kappa_scaling

        return scaling

    def subhalo_mass_function_amplitude(self, sigma_sub, R_ein, zlens):

        """
        :param sigma_sub: units kpc ^ (-2) see Equation 4 in Gilman et al. 2018
        :param m_parent: parent halo mass
        :param zlens: parent halo redshift
        :return:
        """
        #TODO: incorporate redshift dependence

        scaling_factor = self.subhalo_mfunc_spatial_scaing(R_ein, zlens)

        # z_power = 0, until I figure out otherwise
        z_power = 0
        sigma_sub_z = sigma_sub * scaling_factor * (1 + zlens) ** z_power

        return sigma_sub_z

    def norm_A0_from_a0area(self, a0_per_kpc2, zlens, cone_opening_angle, plaw_index, m_pivot = 10**8):

        R_kpc = self.cosmo.kpc_per_asec(zlens) * (0.5 * cone_opening_angle)

        area = numpy.pi * R_kpc ** 2

        return a0_per_kpc2 * m_pivot ** (-plaw_index-1) * area

    def convert_fsub_to_norm(self, fsub, cone_diameter, plaw_index, mlow, mhigh):

        R = cone_diameter*0.5
        area = numpy.pi*R**2

        # convergence is approximately 1/2 at the Einstein radius, this paramterization assumes
        # f_darkmatter >> f_baryon at Einstein radius, so convergence fraction in substructure = 0.5 * fsub
        kappa_sub = 0.5 * fsub

        power = 2 + plaw_index
        integral = power / (mhigh ** power - mlow ** power)

        return area * kappa_sub * self.sigmacrit * integral

    def a0area_main(self, mhalo, z, k1=0.88, k2=1.7, k3=-2):

        # interpolated from galacticus

        logscaling = k1 * numpy.log10(mhalo * 10 ** -13) + k2 * numpy.log10(z + 0.5)

        return 10 ** logscaling

    def projected_mass_fraction(self, sigma_sub, mlow=10**6, mhigh=10**10, mpivot = 10**8, Mhalo = 10**13,
                           z=0.5):

        def _integrand(m, m0):
            x = m * m0 ** -1
            return x*x**-1.9

        sigma_sub *= self.a0area_main(Mhalo, z)
        return sigma_sub * quad(_integrand, mlow, mhigh, args=(mpivot))[0]

    def projected_mass_fraction_arcsec(self, sigma_sub, mlow=10**6, mhigh=10**10, mpivot = 10**8, Mhalo = 10**13,
                           z=0.5):

        mproj = self.projected_mass_fraction(sigma_sub, mlow, mhigh, mpivot, Mhalo, z)

        return mproj*self.cosmo.kpc_per_asec(z) ** 2

    def projected_convergence(self, sigma_sub, mlow=10**6, mhigh=10**10, mpivot = 10**8, Mhalo = 10**13,
                           z=0.5, zsrc=1.5):

        mproj_arcsec = self.projected_mass_fraction_arcsec(sigma_sub, mlow, mhigh, mpivot, Mhalo, z)

        return mproj_arcsec/self.get_sigmacrit_z1z2(z, zsrc)

