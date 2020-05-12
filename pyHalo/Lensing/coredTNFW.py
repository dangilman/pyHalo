from pyHalo.Lensing.numerical_alphas.coreNFWmodifiedtrunc \
    import InterpCNFWmodtrunc, InterpCNFWmodtruncOld
import numpy as np

class coreTNFW(object):

    hybrid = False

    lenstronomy_ID = 'NumericalAlpha'

    def __init__(self, lens_cosmo):

        self.lens_cosmo = lens_cosmo
        self.numerical_class = InterpCNFWmodtrunc()

    def params(self, x, y, mass, redshift, concentration, r_trunc_kpc, b):

        Rs_angle, theta_Rs_nfw = self.lens_cosmo.nfw_physical2angle(mass,
                            concentration, redshift)

        normalization = self._normalize(Rs_angle, theta_Rs_nfw)

        x, y = np.round(x, 4), np.round(y, 4)

        Rs_angle = np.round(Rs_angle, 10)

        r_core = np.round(b * Rs_angle, 10)

        r_trunc = r_trunc_kpc * self.lens_cosmo.cosmo.kpc_per_asec(redshift) ** -1

        kwargs = {'center_x': x, 'center_y': y, 'Rs': Rs_angle,
                  'r_core': r_core, 'norm': normalization, 'r_trunc': r_trunc}

        return kwargs, self.numerical_class

    def _normalize(self, Rs, theta_Rs_nfw):

        bmin = self.numerical_class._betamin
        taumax = self.numerical_class._tau_max

        trs_corenfw = self.numerical_class(Rs, 0, Rs, bmin, taumax, 1)

        norm = theta_Rs_nfw * trs_corenfw ** -1

        return norm


