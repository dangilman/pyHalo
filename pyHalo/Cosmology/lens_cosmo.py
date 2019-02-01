from pyHalo.Cosmology.cosmology import Cosmology
from colossus.halo.concentration import concentration
import numpy
from scipy.integrate import quad

class LensCosmo(object):

    def __init__(self,z_lens,z_source):

        self.cosmo = Cosmology()
        self.z_lens, self.z_source = z_lens, z_source

        # critical density for lensing in units M_sun * Mpc ^ -2
        self.epsilon_crit = self.get_epsiloncrit(z_lens, z_source)
        # critical density for lensing in units M_sun * arcsec ^ -2 at lens redshift
        self.sigmacrit = self.epsilon_crit * (0.001) ** 2 * self.cosmo.kpc_per_asec(z_lens) ** 2
        # critical density of the universe in M_sun Mpc^-3
        self.rhoc = self.cosmo.astropy.critical_density0.value * self.cosmo.density_to_MsunperMpc
        # lensing distances
        self.D_d, self.D_s, self.D_ds = self.cosmo.D_A(0, z_lens), self.cosmo.D_A(0, z_source), self.cosmo.D_A(z_lens, z_source)
        # hubble distance in Mpc
        self._d_hubble = self.cosmo.c * self.cosmo.Mpc * 0.001 * (self.cosmo.h * 100)

        self._kpc_per_asec_zlens = self.cosmo.kpc_per_asec(self.z_lens)

    def mthermal_to_halfmode(self, m_thermal):

        """
        Converts a (fully thermalized) thermal relic particle of mass m [keV] to
        the half-mode mass scale in solar masses (no little h)
        :param m: thermal relic particle mass in keV
        :return: half mode mass in solar masses
        """
        # scaling of 3.3 keV from Viel et al
        norm_h = (2 * 10**8) * 3 ** 3.33 # units M / h
        norm = norm_h / self.cosmo.h

        return norm * m_thermal ** -3.33

    def halfmode_to_thermal(self, m_half_mode):

        """
        Converts a half mode mass in units of solar masses (no little h) to the mass of
        the corresponding thermal relic particle in keV
        :param m: half mode mass in solar masses
        :return: thermal relic particle mass in keV
        """
        norm_h = (2 * 10 ** 8) * 3 ** 3.33  # units M / h
        norm = norm_h / self.cosmo.h

        return (m_half_mode / norm) ** (-1 / 3.33)

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

    def norm_A0_from_a0area(self, a0_per_kpc2, zlens, cone_diameter, plaw_index, m_pivot = 10**8):

        R_kpc = self.cosmo.kpc_per_asec(zlens) * (0.5 * cone_diameter)

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

    def sigmasub_from_fsub(self, fsub, zlens, zsrc, fsub_ref = 0.01):

        scrit_ref = 10**11
        return 0.0135 * fsub * fsub_ref ** -1 * self.get_sigmacrit_z1z2(zlens, zsrc) * scrit_ref ** -1

    def fsub_from_sigmasub(self, sigma_sub, zlens, zsrc, fsub_ref = 0.01, sigma_sub_ref = 0.0135):

        scrit = 10**11
        scrit_z = self.get_sigmacrit_z1z2(zlens, zsrc)
        return sigma_sub * fsub_ref * sigma_sub_ref ** -1 * scrit * scrit_z ** -1

    def mean_mass_fraction(self, mlow, mhigh, mpivot = 10**8, a0_area = 0.0135):

        def _integrand(m, m0):
            x = m * m0 ** -1
            return x*x**-1.9

        return a0_area * quad(_integrand, mlow, mhigh, args=(mpivot))[0]




