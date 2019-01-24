import numpy as np

class cBurkNFWLensing(object):

    hybrid = True

    def __init__(self, lens_cosmo = None, zlens = None, z_source = None):

        if lens_cosmo is None:
            from pyHalo.Cosmology.lens_cosmo import LensCosmo
            lens_cosmo = LensCosmo(zlens, z_source)

        self.lens_cosmo = lens_cosmo

    def _interpolating_function(self, rs, r_core, b_min = 0.3,
                 b_max = 0.7, b_crit = 0.5, c = 1):

        beta = r_core * rs ** -1

        b_half = 0.5 * (b_max - b_min)

        arg = (beta - b_crit) * b_half ** -1

        return 0.5 * (1 + np.tanh(c * arg))

    def _transform(self, theta_Rs_nfw, theta_Rs_cburk, f):

        return f*theta_Rs_cburk, (1-f)*theta_Rs_nfw

    def params(self, x, y, mass, concentration, q, redshift):

        rs_nfw, trs_nfw = self.lens_cosmo.nfw_physical2angle(mass, concentration, redshift)

        rs, trs, r_core = self.lens_cosmo.coreBurkert_physical2angle(mass,
                                     concentration, redshift, q)

        f = self._interpolating_function(rs_nfw, r_core)

        trs, trs_nfw = self._transform(trs_nfw, trs, f)

        kwargs1 = {'theta_Rs': trs_nfw, 'Rs': rs, 'center_x': x, 'center_y':y}
        kwargs2 = {'theta_Rs': trs, 'Rs': rs, 'r_core': r_core,'center_x': x, 'center_y':y}

        return [kwargs1, kwargs2]

    def M_physical(self, m, c, z):
        """
        :param m200: m200
        :return: physical mass corresponding to m200
        """
        rho0, Rs, r200 = self.lens_cosmo.NFW_params_physical(m,c,z)
        return 4*np.pi*rho0*Rs**3*(np.log(1+c)-c*(1+c)**-1)
