from pyHalo.Halos.cosmo_profiles import CosmoMassProfiles
import numpy
from scipy.integrate import quad

class TNFW(CosmoMassProfiles):

    def tnfw_mass_2angles(self, M, r_t, z):

        c = self.NFW_concentration(M, z, scatter=False)

        Rs_angle, theta_Rs = self.tnfw_physical2angle(M, c, r_t, z)

        return Rs_angle, theta_Rs

    def tnfw_physical2angle(self, M, c, r_t, z):
        """
        converts the physical mass and concentration parameter of an NFW profile into the lensing quantities
        :param M: mass enclosed 200 \rho_crit
        :param c: NFW concentration parameter (r200/r_s)
        :return: theta_Rs (observed bending angle at the scale radius, Rs_angle (angle at scale radius) (in units of arcsec)
        """
        D_d = self.lens_cosmo.cosmo.D_A(0, z)
        rho0, Rs, r200 = self._nfwParam_physical_Mpc(M, c, z)

        Rs_angle = Rs / D_d / self.lens_cosmo.cosmo.arcsec  # Rs in arcsec
        #theta_Rs = rho0 * (4 * Rs ** 2 * (1 + numpy.log(1. / 2.)))
        theta_Rs = self.tnfw_theta_Rs(Rs, rho0, r_t * Rs_angle ** -1)
        eps_crit = self.lens_cosmo.get_epsiloncrit(z, self.lens_cosmo.z_source)
        return Rs_angle, theta_Rs / eps_crit / D_d / self.lens_cosmo.cosmo.arcsec

    def tnfw_theta_Rs(self, rs, rho, tau):

        factor = self._tnfw_g1(tau)

        theta_Rs = rho * 4 * rs ** 2 * factor
        return theta_Rs

    def _tnfw_g1(self, tau):

        x = 1
        L = numpy.log((numpy.sqrt(tau**2 + 1)+tau)**-1)
        F = 1

        return tau ** 2 * (tau ** 2 + 1) ** -2 * (
            (tau ** 2 + 1 + 2 * (x ** 2 - 1)) * F + tau * numpy.pi + (tau ** 2 - 1) * numpy.log(tau) +
            numpy.sqrt(tau ** 2 + x ** 2) * (-numpy.pi + L * (tau ** 2 - 1) * tau ** -1))



