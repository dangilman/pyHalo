from pyHalo.Halos.cosmo_profiles import CosmoMassProfiles
import numpy

class modifiedCNFW(CosmoMassProfiles):

    def modcorenfw_physical2angle(self, M, c, z):
        """
        converts the physical mass and concentration parameter of an NFW profile into a scale radius and a central density
        :param M: mass enclosed 200 \rho_crit
        :param c: NFW concentration parameter (r200/r_s)
        :return: theta_Rs (observed bending angle at the scale radius, Rs_angle (angle at scale radius) (in units of arcsec)
        """
        D_d = self.lens_cosmo.cosmo.D_A(0, z)
        rho0, Rs, r200 = self._nfwParam_physical_Mpc(M, c, z)

        Rs_angle = Rs / D_d / self.lens_cosmo.cosmo.arcsec  # Rs in arcsec

        eps_crit = self.lens_cosmo.get_epsiloncrit(z, self.lens_cosmo.z_source)

        return Rs_angle, rho0 / eps_crit / D_d / self.lens_cosmo.cosmo.arcsec

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
