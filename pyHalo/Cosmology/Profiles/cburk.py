from pyHalo.Cosmology.Profiles.cosmo_profiles import CosmoMassProfiles
import numpy

class CBURK(CosmoMassProfiles):

    def rho0_c_CBURK(self, c, b):
        """
        computes density normalization as a function of concentration parameter
        :return: density normalization in h^2/Mpc^3 (comoving)
        """

        func = 2*(1+b**2) ** -1 * (0.5*numpy.log(1+c**2) + b**2*numpy.log(c*b**-1 + 1) -
                                 b*numpy.arctan(c))

        return 200. / 3 * self.lens_cosmo.rhoc * c ** 3 / func

    def _cburkParam_physical_Mpc(self, M, c, z, b):

        """
        returns the NFW parameters in physical units
        :param M: physical mass in M_sun
        :param c: concentration
        :return:
        """

        h = self.lens_cosmo.cosmo.h
        a_z = self.lens_cosmo.cosmo.scale_factor(z)

        r200 = self.rN_M_nfw(M * h, 200) * a_z / h  # physical radius r200
        rho0 = self.rho0_c_CBURK(c, b) * h ** 2 / a_z ** 3  # physical density in M_sun/Mpc**3
        Rs = r200 / c
        return rho0, Rs, r200

    def _cburk_params_physical_Mpc(self, M, c, z, b):

        """
        :param M: m200
        :param c: concentration
        :param z: redshift
        :param b: ratio of core radius to Rs, b * Rs = r_core
        :return:
        """

        rho_nfw, Rs, r200 = self._cburkParam_physical_Mpc(M, c, z, b)

        r_core = b * Rs

        return rho_nfw, Rs, r_core, r200

    def cburk_physical2angle(self, M, c, z, b):

        """

        :param M:
        :param c:
        :param z:
        :param q: Rs / r_core
        :return:
        """

        arcsec = self.lens_cosmo.cosmo.arcsec

        D_d = self.lens_cosmo.cosmo.D_A(0, z)
        rho0, Rs, r_core, r200 = self._cburk_params_physical_Mpc(M, c, z, b)

        Rs_angle = Rs * (D_d * arcsec) ** -1  # Rs in arcsec
        r_core_angle = r_core * (D_d * arcsec) ** -1  # r_core in arcsec

        # note the definition of p in self.cburk_alpha
        p = Rs * r_core ** -1
        def_Rs = 4 * Rs ** 2 * self.cburk_alpha(1, p) * rho0

        eps_crit = self.lens_cosmo.get_epsiloncrit(z, self.lens_cosmo.z_source)
        theta_Rs = def_Rs * (eps_crit * D_d * arcsec) ** -1

        return Rs_angle, theta_Rs, r_core_angle

    def cburk_alpha(self, x, p):

        prefactor = (1 + p ** 2) ** -1

        ux = numpy.sqrt(1 + x**2)

        if x * p == 1:

            func = numpy.log(0.25 * x ** 2 * p ** 2) + numpy.pi * p * (ux - 1) + \
                   2 * p ** 2 * (ux * numpy.arctanh(ux ** -1) +
                                 numpy.log(0.5 * x))

        elif x * p < 1:

            gx = (1 - x**2*p**2)**0.5
            func = numpy.log(0.25 * x ** 2 * p ** 2) + numpy.pi * p * (ux - 1) + \
                   2 * p ** 2 * (ux * numpy.arctanh(ux ** -1) +
                                 numpy.log(0.5 * x)) + 2 * gx * numpy.arctanh(gx)

        else:
            fx = (x ** 2 * p ** 2 - 1) ** 0.5
            func = numpy.log(0.25 * x ** 2 * p ** 2) + numpy.pi * p * (ux - 1) + \
                   2 * p ** 2 * (ux * numpy.arctanh(ux ** -1) +
                                 numpy.log(0.5 * x)) - 2 * fx * numpy.arctan(fx)

        return func * prefactor

