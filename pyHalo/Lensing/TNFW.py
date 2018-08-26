from pyHalo.Lensing.NFW import NFWLensing
import numpy as np

class TNFWLensing(object):

    def __init__(self,lens_cosmo):

        self.lens_cosmo = lens_cosmo

    def params(self, x, y, mass, concentration, redshift, r_trunc):

        theta_Rs, Rs_angle = self.nfw_physical2angle(mass, concentration, redshift)

        kwargs = {'theta_Rs':theta_Rs, 'Rs': Rs_angle,
                  'center_x':x, 'center_y':y, 'r_trunc':r_trunc}

        return kwargs

    def mass_finite(self, m200, c, z, r_trunc):

        rho, Rs, r200 = self.lens_cosmo.NFW_params_physical(m200, c, z)

        Rs_arcsec = Rs * self.lens_cosmo.cosmo.kpc_per_asec(z) ** -1
        tau = r_trunc * Rs_arcsec ** -1
        t2 = tau ** 2

        return 4 * np.pi * Rs ** 3 * rho * t2 * (t2 + 1) ** -2 * (
                (t2 - 1) * np.log(tau) + np.pi * tau - (t2 + 1))

    def nfw_physical2angle(self, m, c, z):
        """
        converts the physical mass and concentration parameter of an NFW profile into the lensing quantities
        :param M: mass enclosed 200 \rho_crit
        :param c: NFW concentration parameter (r200/r_s)
        :return: theta_Rs (observed bending angle at the scale radius, Rs_angle (angle at scale radius) (in units of arcsec)
        """

        rho0, Rs, r200 = self.lens_cosmo.NFW_params_physical(m,c,z)
        rho0_mpc = rho0 * 1000**3
        Rs_mpc = Rs * 0.001

        Rs_angle = Rs_mpc / self.lens_cosmo.cosmo.D_A(0,z) / self.lens_cosmo.cosmo.arcsec #Rs in asec

        tRs = rho0_mpc * (4 * Rs_mpc ** 2 * (1 + np.log(1. / 2.)))

        eps_crit = self.lens_cosmo.get_epsiloncrit(z,self.lens_cosmo.z_source)
        dA = self.lens_cosmo.cosmo.D_A(0, z)

        theta_Rs = tRs / eps_crit / dA / self.lens_cosmo.cosmo.arcsec

        return theta_Rs, Rs_angle
