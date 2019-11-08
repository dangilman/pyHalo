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

    def __init__(self, z_lens, z_source, cosmology):

        self.cosmo = cosmology
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

    @property
    def colossus(self):
        return self.cosmo.colossus

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
        D_d = self.cosmo.D_A_z(z1)
        D_s = self.cosmo.D_A_z(z2)

        epsilon_crit = (self.cosmo.c**2*(4*numpy.pi*self.cosmo.G)**-1)*(D_s*D_ds**-1*D_d**-1)

        return epsilon_crit

    def get_sigmacrit(self, z):

        return self.get_epsiloncrit(z,self.z_source)*(0.001)**2*self.cosmo.kpc_per_asec(z)**2

    def get_sigmacrit_z1z2(self,zlens,zsrc):

        return self.get_epsiloncrit(zlens,zsrc)*(0.001)**2*self.cosmo.kpc_per_asec(zlens)**2

    def norm_A0_from_a0area(self, a0_per_kpc2, zlens, cone_opening_angle, plaw_index, m_pivot = 10**8):

        R_kpc = self.cosmo.kpc_per_asec(zlens) * (0.5 * cone_opening_angle)

        area = numpy.pi * R_kpc ** 2

        return a0_per_kpc2 * m_pivot ** (-plaw_index-1) * area

    def convert_fsub_to_norm(self, mass_in_subhalos, cone_opening_angle, zlens, plaw_index, mlow,
                             mhigh, mpivot=10**8):

        power = 2+plaw_index
        R_kpc = self.cosmo.kpc_per_asec(zlens) * (0.5 * cone_opening_angle)
        area = numpy.pi * R_kpc ** 2
        integral = (mpivot/power) * ((mhigh/mpivot)**power - (mlow/mpivot)**power)
        sigma_sub =  mass_in_subhalos / integral / area

        return sigma_sub
