import numpy as np

class cBurkLensing(object):

    hybrid = False

    def __init__(self, lens_cosmo = None, zlens = None, z_source = None):

        if lens_cosmo is None:
            from pyHalo.Cosmology.lens_cosmo import LensCosmo
            lens_cosmo = LensCosmo(zlens, z_source)

        self.lens_cosmo = lens_cosmo

    def params(self, x, y, mass, concentration, q, redshift):

        rs, trs, r_core = self.lens_cosmo.coreBurkert_physical2angle(mass,
                                     concentration, redshift, q)

        kwargs = {'theta_Rs': trs, 'Rs': rs, 'r_core': r_core,
                  'center_x':x, 'center_y':y}

        return kwargs

    def M_physical(self, m, c, z):
        """
        :param m200: m200
        :return: physical mass corresponding to m200
        """
        rho0, Rs, r200 = self.lens_cosmo.NFW_params_physical(m,c,z)
        return 4*np.pi*rho0*Rs**3*(np.log(1+c)-c*(1+c)**-1)
