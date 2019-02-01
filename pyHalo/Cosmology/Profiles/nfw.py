from pyHalo.Cosmology.Profiles.cosmo_profiles import CosmoMassProfiles
import numpy

class NFW(CosmoMassProfiles):

    def rho0_c_NFW(self, c):
        """
        computes density normalization as a function of concentration parameter
        :return: density normalization in h^2/Mpc^3 (comoving)
        """

        return 200. / 3 * self.lens_cosmo.rhoc * c ** 3 / (numpy.log(1 + c) - c / (1 + c))

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

    def _nfwParam_physical_Mpc(self, M, c, z):

        """
        returns the NFW parameters in physical units
        :param M: physical mass in M_sun
        :param c: concentration
        :return:
        """

        h = self.lens_cosmo.cosmo.h
        a_z = self.lens_cosmo.cosmo.scale_factor(z)

        r200 = self.rN_M_nfw(M * h, 200) * a_z / h   # physical radius r200
        rho0 = self.rho0_c_NFW(c) * h ** 2 / a_z ** 3 # physical density in M_sun/Mpc**3
        Rs = r200/c
        return rho0, Rs, r200

    def NFW_params_physical(self, M, c, z):

        rho0, Rs, r200 = self._nfwParam_physical_Mpc(M, c, z)

        return rho0 * 1000 ** -3, Rs * 1000, r200 * 1000

    def LOS_truncation(self, M, z, N=50):
        """
        Truncate LOS halos at r50
        :param M:
        :param c:
        :param z:
        :param N:
        :return:
        """
        # in mega-parsec
        r50_mpc = self.rN_M_nfw(M * self.lens_cosmo.cosmo.h, N) * (self.lens_cosmo.cosmo.h * (1 + z)) ** -1

        # in kpc
        r50 = r50_mpc * 1000
        r50_asec = r50 * self.lens_cosmo.cosmo.kpc_per_asec(z) ** -1

        return r50_asec
