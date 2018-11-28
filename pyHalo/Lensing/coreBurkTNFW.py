import numpy as np

class cBurkNFWLensing(object):

    hybrid = True

    def __init__(self, lens_cosmo = None, zlens = None, z_source = None,
                 interp_coeff = 1, interp_power = 2, interp_qcrit = 0.2):

        if lens_cosmo is None:
            from pyHalo.Cosmology.lens_cosmo import LensCosmo
            lens_cosmo = LensCosmo(zlens, z_source)

        self.lens_cosmo = lens_cosmo

        self._interp_coeff = interp_coeff
        self._interp_power = interp_power
        self._interp_qcrit = interp_qcrit

    def _interpolating_function(self, rs, r_core):

        ratio = (r_core * rs**-1) * self._interp_qcrit**-1
        arg = self._interp_coeff*(ratio)**self._interp_power
        f = np.exp(-arg)

        return f

    def _transform(self, theta_Rs_nfw, theta_Rs_cburk, f):

        return f*theta_Rs_nfw, (1-f)*theta_Rs_cburk

    def params(self, x, y, mass, concentration, q, redshift):

        rs, trs, r_core = self.lens_cosmo.coreBurkert_physical2angle(mass,
                                     concentration, redshift, q)

        rs_nfw, trs_nfw = self.lens_cosmo.nfw_physical2angle(mass, concentration, redshift)

        f = self._interpolating_function(rs, r_core)
        trs_nfw, trs = self._transform(trs_nfw, trs, f)
        #rescale = self.lens_cosmo.rescale_rho_burk(mass, rho_nfw, q ** -1, rs, concentration)

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
