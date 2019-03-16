import numpy as np
from pyHalo.Halos.Profiles.tnfw import TNFW
from pyHalo.Lensing.numerical_alphas.coreNFWmodifiedtrunc import InterpCNFWmodtrunc

class coreNFWmodifiedtruncLensing(object):

    hybrid = False

    lenstronomy_ID = 'NumericalAlpha'

    def __init__(self, lens_cosmo = None, zlens = None, z_source = None):

        if lens_cosmo is None:
            from pyHalo.Cosmology.lens_cosmo import LensCosmo
            lens_cosmo = LensCosmo(zlens, z_source)

        self.lens_cosmo = TNFW(lens_cosmo)
        self.numerical_class = InterpCNFWmodtrunc()

    def params(self, x, y, mass, concentration, b, redshift, r_trunc):

        Rs_angle, theta_Rs_nfw = self.lens_cosmo.tnfw_physical2angle(mass,
                       concentration, r_trunc, redshift)

        _, normalization = self._compute_properties(Rs_angle, theta_Rs_nfw)

        r_core = b*Rs_angle
        kwargs = {'center_x': x, 'center_y': y,'Rs': Rs_angle,
                  'r_core': r_core, 'norm': normalization, 'r_trunc': r_trunc}

        return kwargs, self.numerical_class

    def _compute_properties(self, Rs, theta_Rs_nfw):

        bmin = self.numerical_class._betamin
        taumax = self.numerical_class._tau_max

        trs_corenfw = self.numerical_class(Rs, 0, Rs, bmin, taumax, 1)

        norm = theta_Rs_nfw * trs_corenfw ** -1
        r_core = 0.5*Rs
        return r_core, norm


