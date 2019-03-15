from pyHalo.Halos.cosmo_profiles import CosmoMassProfiles
import numpy
from scipy.integrate import quad

class NFW(CosmoMassProfiles):

    def vmax(self, M, z):

        pass

    def mean_circular_velocity(self, M, z, rmax=None):
        """
        mean circular velocity inside rmax for an NFW halo
        :param M:
        :param z:
        :param rmax:
        :return:
        """
        c = self.NFW_concentration(M, z, scatter=False)

        rho0, rs, r200 = self.NFW_params_physical(M, c, z)

        G = 4.3e-3 # pc/Msun (km/sec)^2
        G *= 0.001 # kpc/Msun (km/sec)^2
        factor = numpy.sqrt(4*numpy.pi*rs**3*rho0 * G)

        def _integrand(r, Rs):
            x = r * Rs ** -1
            return numpy.sqrt(r**-1*(numpy.log(1+x) - x * (x+1)**-1))

        if rmax is None:
            rmax = rs

        factor *= rmax ** -1

        return factor * quad(_integrand, 0.0001*rs, rmax, args=(rs))[0]

    def nfw_mass_2angles(self, M, z):

        c = self.NFW_concentration(M, z, scatter=False)

        Rs_angle, theta_Rs = self.nfw_physical2angle(M, c, z)

        return Rs_angle, theta_Rs

    def nfw_physical2angle(self, M, c, z):
        """
        converts the physical mass and concentration parameter of an NFW profile into the lensing quantities
        :param M: mass enclosed 200 \rho_crit
        :param c: NFW concentration parameter (r200/r_s)
        :return: theta_Rs (observed bending angle at the scale radius, Rs_angle (angle at scale radius) (in units of arcsec)
        """
        D_d = self.lens_cosmo.cosmo.D_A(0, z)
        rho0, Rs, r200 = self._nfwParam_physical_Mpc(M, c, z)

        Rs_angle = Rs / D_d / self.lens_cosmo.cosmo.arcsec  # Rs in arcsec
        theta_Rs = rho0 * (4 * Rs ** 2 * (1 + numpy.log(1. / 2.)))
        eps_crit = self.lens_cosmo.get_epsiloncrit(z, self.lens_cosmo.z_source)
        return Rs_angle, theta_Rs / eps_crit / D_d / self.lens_cosmo.cosmo.arcsec



