from pyHalo.Halos.cosmo_profiles import CosmoMassProfiles
import numpy
from pyHalo.Halos.Profiles.rcore_interpolation import linear_term, constant_term
from scipy.interpolate import interp2d

class coreTNFW(CosmoMassProfiles):

    rho_bins = numpy.array([5 * 10 ** 6, 10 ** 7, 5 * 10 ** 7, 10 ** 8, 5 * 10 ** 8])
    rs_bins = numpy.array(10 ** numpy.array([-0.75, -0.4, -0.25, 0.0, 0.25, 0.50]))
    log_rs_arr, log_rho_arr = numpy.meshgrid(numpy.log10(rs_bins), numpy.log10(rho_bins))
    linear_interp = interp2d(log_rho_arr, log_rs_arr, linear_term)
    constant_interp = interp2d(log_rho_arr, log_rs_arr, constant_term)

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

    def rcore_from_rhors(self, rho_s, Rs, zeta):

        ratio = self.rcore_over_rs_from_rhors(rho_s, Rs, zeta)

        return ratio * Rs

    def rcore_over_rs_from_rhors(self, rho_s, Rs, zeta):

        log_rho_value = numpy.log10(rho_s)
        log_rs_value = numpy.log10(Rs)
        log_zeta_value = numpy.log10(zeta)

        p0 = self.linear_interp(log_rho_value, log_rs_value)
        p1 = self.constant_interp(log_rho_value, log_rs_value)

        rho_sidm = 10 ** (p0 * log_zeta_value + p1)
        return 10 ** log_rho_value * rho_sidm ** -1





