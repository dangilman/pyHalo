import numpy as np
from pyHalo.Rendering.parameterizations import BrokenPowerLaw
from pyHalo.Spatial.nfw import NFW3D
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Rendering.keywords import subhalo_mass_function
from pyHalo.Spatial.keywords import subhalo_spatial_NFW
from pyHalo.Rendering.Main.mainlens import MainLensPowerLaw

class MainLensPowerLawDynamic(object):

    def __init__(self, args, geometry, x_center_lens, y_center_lens,
                 x_aperture, y_aperture, aperture_size):

        self.main = MainLensPowerLaw(args, geometry, x_center_lens, y_center_lens)

        self._center_x, self._center_y = x_aperture, y_aperture
        self._rmax = aperture_size

    def __call__(self):
        """

        :return: x coordinate, y coordinates, r3d, r3d
        NOTE: x and y are returned in arcsec, while r2d and r3d are expressed in kpc
        """

        x_arcsec, y_arcsec, r2d_kpc, r3d_kpc, _ = self.main()

        dx = x_arcsec - self._center_x
        dy = y_arcsec - self._center_y
        dr = np.sqrt(dx ** 2 + dy ** 2)
        inds = np.where(dr < self._rmax)

        redshifts = [self.main._geometry._zlens] * len(dx[inds])

        return x_arcsec[inds], y_arcsec[inds], r2d_kpc[inds], r3d_kpc[inds], redshifts

