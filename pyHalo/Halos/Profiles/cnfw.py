from pyHalo.Halos.cosmo_profiles import CosmoMassProfiles
import numpy

class CNFW(CosmoMassProfiles):

    def rho0_c_CNFW(self, c, b):
        """
        computes density normalization as a function of concentration parameter
        :return: density normalization in h^2/Mpc^3 (comoving)
        """

        func = (c * (1+c) ** -1 * (-1+b) ** -1 + (-1+b) ** -2 *
                      ((2*b-1)*numpy.log(1/(1+c)) + b **2 * numpy.log(c / b + 1)))

        return 200. / 3 * self.lens_cosmo.rhoc * c ** 3 / func

    def _cnfwParam_physical_Mpc(self, M, c, z, b):

        """
        returns the NFW parameters in physical units
        :param M: physical mass in M_sun
        :param c: concentration
        :return:
        """

        h = self.lens_cosmo.cosmo.h
        a_z = self.lens_cosmo.cosmo.scale_factor(z)

        r200 = self.rN_M_nfw_comoving(M * h, 200) * a_z / h   # physical radius r200
        rho0 = self.rho0_c_CNFW(c, b) * h ** 2 / a_z ** 3 # physical density in M_sun/Mpc**3
        Rs = r200/c
        return rho0, Rs, r200

    def _corednfw_params_physical_Mpc(self, M, c, z, b):

        """
        :param M: m200
        :param c: concentration
        :param z: redshift
        :param b: ratio of core radius to Rs, b * Rs = r_core
        :return:
        """

        rho_nfw, Rs, r200 = self._cnfwParam_physical_Mpc(M, c, z, b)

        r_core = b * Rs

        return rho_nfw, Rs, r_core, r200

    def corenfw_physical2angle(self, M, c, z, b):

        """

        :param M:
        :param c:
        :param z:
        :param q: Rs / r_core
        :return:
        """

        arcsec = self.lens_cosmo.cosmo.arcsec

        D_d = self.lens_cosmo.cosmo.D_A(0, z)
        rho0, Rs, r_core, r200 = self._corednfw_params_physical_Mpc(M, c, z, b)

        Rs_angle = Rs * (D_d * arcsec) ** -1  # Rs in arcsec
        r_core_angle = r_core * (D_d * arcsec) ** -1  # r_core in arcsec
        def_Rs = 4 * Rs ** 2 * self.corenfw_alpha(1, r_core * Rs ** -1) * rho0

        eps_crit = self.lens_cosmo.get_epsiloncrit(z, self.lens_cosmo.z_source)
        theta_Rs = def_Rs * (eps_crit * D_d * arcsec) ** -1

        return Rs_angle, theta_Rs, r_core_angle

    def corenfw_alpha(self, x, b):

        def _nfw_func(y):
            if y < 1:
                a = numpy.sqrt(1-y**2)
                return numpy.arctanh(a) * a ** -1
            elif y == 1:
                return 1
            else:
                a = numpy.sqrt(-1 + y ** 2)
                return numpy.arctan(a) * a ** -1

        if b == 1:
            b = 1+0.0001

        b2 = b**2
        x2 = x**2

        fac = (1-b) ** 2
        prefac = fac ** -1

        if numpy.absolute(x-b) < 0.0001:
            output = prefac * (2*(1-2*b+b**3) * _nfw_func(b) + \
                            fac * (-1.38692 + numpy.log(b2)) - b2*numpy.log(b2))
        else:
            output = prefac * (fac * numpy.log(0.25 * x2) - b2 * numpy.log(b2) + \
                2 * (b2 - x2) * _nfw_func(x * b**-1) + 2 * (1+b*(x2 - 2))*
                             _nfw_func(x))

        return 0.5*output

