import numpy as np

class NFWLensing(object):

    def __init__(self, lens_cosmo):
        self.lens_cosmo = lens_cosmo

    def params(self, x, y, mass, concentration, redshift):

        theta_Rs, Rs_angle = self.nfw_physical2angle(mass, concentration, redshift)

        kwargs = {'theta_Rs':theta_Rs, 'Rs': Rs_angle,
                  'center_x':x, 'center_y':y}

        return kwargs

    def nfw_physical2angle(self, m, c, z):
        """
        converts the physical mass and concentration parameter of an NFW profile into the lensing quantities
        :param M: mass enclosed 200 \rho_crit
        :param c: NFW concentration parameter (r200/r_s)
        :return: theta_Rs (observed bending angle at the scale radius, Rs_angle (angle at scale radius) (in units of arcsec)
        """

        if z < 1e-4:
            z = 1e-4

        rho0, Rs, r200 = self.lens_cosmo.NFW_params_physical(m,c,z)
        rho0_mpc = rho0 * 1000**3
        Rs_mpc = Rs * 0.001

        Rs_angle = Rs_mpc / self.lens_cosmo.cosmo.D_A(0,z) / self.lens_cosmo.cosmo.arcsec #Rs in asec

        tRs = rho0_mpc * (4 * Rs_mpc ** 2 * (1 + np.log(1. / 2.)))

        eps_crit = self.lens_cosmo.get_epsiloncrit(z,self.lens_cosmo.z_source)
        dA = self.lens_cosmo.cosmo.D_A(0, z)

        theta_Rs = tRs / eps_crit / dA / self.lens_cosmo.cosmo.arcsec

        return theta_Rs, Rs_angle

    def M_physical(self, m, c, z):
        """

        :param m200: m200
        :return: physical mass corresponding to m200
        """

        rho0, Rs, r200 = self.lens_cosmo.NFW_params_physical(m,c,z)
        return 4*np.pi*rho0*Rs**3*(np.log(1+c)-c*(1+c)**-1)
