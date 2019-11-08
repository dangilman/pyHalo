from pyHalo.Halos.cosmo_profiles import CosmoMassProfiles
import numpy
from scipy.integrate import quad
from pyHalo.Halos.Profiles.tnfw import TNFW

class NFW(CosmoMassProfiles):

    def vmax(self, M, z):

        pass

    def nfw_mass_2angles(self, M, z, model='diemer19'):

        c = self.NFW_concentration(M, z, scatter=False, model=model)

        Rs_angle, theta_Rs = self.nfw_physical2angle(M, c, z)

        return Rs_angle, theta_Rs

    def nfw_physical2angle(self, M, c, z):
        """
        converts the physical mass and concentration parameter of an NFW profile into the lensing quantities
        :param M: mass enclosed 200 \rho_crit
        :param c: NFW concentration parameter (r200/r_s)
        :return: theta_Rs (observed bending angle at the scale radius, Rs_angle (angle at scale radius) (in units of arcsec)
        """
        D_d = self.lens_cosmo.cosmo.D_A_z(z)
        rho0, Rs, r200 = self._nfwParam_physical_Mpc(M, c, z)

        Rs_angle = Rs / D_d / self.lens_cosmo.cosmo.arcsec  # Rs in arcsec
        theta_Rs = rho0 * (4 * Rs ** 2 * (1 + numpy.log(1. / 2.)))
        eps_crit = self.lens_cosmo.get_epsiloncrit(z, self.lens_cosmo.z_source)
        return Rs_angle, theta_Rs / eps_crit / D_d / self.lens_cosmo.cosmo.arcsec
