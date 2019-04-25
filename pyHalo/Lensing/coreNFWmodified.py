import numpy as np
from pyHalo.Halos.Profiles.nfw import NFW
from pyHalo.Lensing.numerical_alphas.coreNFWmodified import InterpCNFWmod

class coreNFWmodifiedLensing(object):

    hybrid = False

    lenstronomy_ID = 'NumericalAlpha'

    def __init__(self, lens_cosmo = None, zlens = None, z_source = None):

        if lens_cosmo is None:
            from pyHalo.Cosmology.lens_cosmo import LensCosmo
            lens_cosmo = LensCosmo(zlens, z_source)

        self.lens_cosmo = NFW(lens_cosmo)
        self.numerical_class = InterpCNFWmod()

    def params(self, x, y, mass, concentration, b, redshift):

        Rs_angle, theta_Rs_nfw = self.lens_cosmo.nfw_physical2angle(mass,
                       concentration, redshift)

        _, normalization = self._compute_properties(Rs_angle, theta_Rs_nfw, redshift)

        x, y = np.round(x, 4), np.round(y, 4)

        Rs_angle = np.round(Rs_angle, 6)

        r_core = b * Rs_angle
        kwargs = {'center_x': x, 'center_y': y,'Rs': Rs_angle,
                  'r_core': r_core, 'norm': normalization}

        return kwargs, self.numerical_class

    def _compute_properties(self, rs_arcsec, theta_Rs_nfw, z):

        trs_corenfw = self.numerical_class(rs_arcsec, 0, rs_arcsec,
                                           self.numerical_class._betamin, 1)

        norm = theta_Rs_nfw * trs_corenfw ** -1
        r_core = 0.5*rs_arcsec
        return r_core, norm


