import numpy as np

class cBurkTNFWLensing(object):

    hybrid = True

    def __init__(self, lens_cosmo = None, zlens = None, z_source = None):

        if lens_cosmo is None:
            from pyHalo.Cosmology.lens_cosmo import LensCosmo
            lens_cosmo = LensCosmo(zlens, z_source)

        self.lens_cosmo = lens_cosmo

    def _interpolating_function(self, kwargs1, kwargs2):

        rs = kwargs1['Rs']
        r_core = kwargs2['r_core']

        power = 4
        coeff = 1
        q_crit = 0.2
        ratio = (r_core * rs**-1) * q_crit**-1
        arg = coeff*(ratio)**power
        f = np.exp(-arg)

        return f

    def _transform(self, theta_Rs_nfw, theta_Rs_cburk, f):

        return f*theta_Rs_nfw, (1-f)*theta_Rs_cburk

    def params(self, x, y, mass, concentration, q, r_trunc, redshift):

        rs, trs, r_core = self.lens_cosmo.coreBurkert_physical2angle(mass,
                                     concentration, redshift, q)

        rs_nfw, trs_nfw = self.lens_cosmo.nfw_physical2angle(mass, concentration, redshift)

        kwargs1 = {'theta_Rs': trs_nfw, 'Rs': rs, 'r_trunc': r_trunc}
        kwargs2 = {'theta_Rs': trs, 'Rs': rs, 'r_core': r_core}

        kwargs = {'kwargs1': kwargs1, 'kwargs2': kwargs2}

        return kwargs, {'lens_model_1': 'TNFW', 'lens_model_2': 'coreBURKERT',
                        'interpolating_function':self._interpolating_function}

    def M_physical(self, m, c, z):
        """
        :param m200: m200
        :return: physical mass corresponding to m200
        """
        rho0, Rs, r200 = self.lens_cosmo.NFW_params_physical(m,c,z)
        return 4*np.pi*rho0*Rs**3*(np.log(1+c)-c*(1+c)**-1)
