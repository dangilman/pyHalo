import numpy as np
from pyHalo.Lensing.coreBurk import cBurkLensing
from pyHalo.Lensing.coreNFW import coreNFWLensing

class cBurkcNFWLensing(object):

    hybrid = True

    def __init__(self, lens_cosmo = None, zlens = None, z_source = None):

        if lens_cosmo is None:
            from pyHalo.Cosmology.lens_cosmo import LensCosmo
            lens_cosmo = LensCosmo(zlens, z_source)

        self.lens_cosmo = lens_cosmo

        self._cburk = cBurkLensing(lens_cosmo)
        self._cnfw = coreNFWLensing(lens_cosmo)

    def _interpolating_function(self, rs, r_core, b_min = 0.3,
                 b_max = 0.7, b_crit = 0.5, c = 1):

        beta = r_core * rs ** -1

        b_half = 0.5 * (b_max - b_min)

        arg = (beta - b_crit) * b_half ** -1

        return 0.5 * (1 + np.tanh(c * arg))

    def params(self, x, y, mass, concentration, b, redshift):

        kwargs_cburk = self._cburk.params(x, y, mass, concentration, b, redshift)
        kwargs_cnfw = self._cnfw.params(x, y, mass, concentration, b, redshift)

        f = self._interpolating_function(kwargs_cnfw['Rs'], kwargs_cnfw['r_core'])
        #f = 1
        kwargs_cburk['theta_Rs'] = kwargs_cburk['theta_Rs'] * f
        kwargs_cnfw['theta_Rs'] = kwargs_cnfw['theta_Rs'] * (1-f)

        return [kwargs_cnfw, kwargs_cburk]

    def M_physical(self, m, c, z):
        """
        :param m200: m200
        :return: physical mass corresponding to m200
        """
        rho0, Rs, r200 = self.lens_cosmo.NFW_params_physical(m,c,z)
        return 4*np.pi*rho0*Rs**3*(np.log(1+c)-c*(1+c)**-1)
