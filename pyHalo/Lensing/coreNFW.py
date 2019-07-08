import numpy as np
from pyHalo.Halos.Profiles.cnfw import CNFW

class coreNFWLensing(object):

    hybrid = False

    lenstronomy_ID = 'CNFW'

    def __init__(self, lens_cosmo = None, zlens = None, z_source = None):

        if lens_cosmo is None:
            from pyHalo.Cosmology.lens_cosmo import LensCosmo
            lens_cosmo = LensCosmo(zlens, z_source)

        self.lens_cosmo = CNFW(lens_cosmo)

    def params(self, x, y, mass, redshift, concentration, b):

        Rs_angle, theta_Rs, r_core = self.lens_cosmo.corenfw_physical2angle(mass,
                       concentration, redshift, b)

        x, y = np.round(x, 4), np.round(y, 4)

        Rs_angle = np.round(Rs_angle, 6)
        theta_Rs = np.round(theta_Rs, 6)

        kwargs = {'alpha_Rs':theta_Rs, 'Rs': Rs_angle, 'r_core': r_core,
                  'center_x':x, 'center_y':y}

        return kwargs, None

    def M_physical(self, m, c, z):
        """
        :param m200: m200
        :return: physical mass corresponding to m200
        """
        rho0, Rs, r200 = self.lens_cosmo.NFW_params_physical(m,c,z)
        return 4*np.pi*rho0*Rs**3*(np.log(1+c)-c*(1+c)**-1)
