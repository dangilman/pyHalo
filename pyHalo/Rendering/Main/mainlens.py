import numpy as np
from pyHalo.Rendering.parameterizations import BrokenPowerLaw
from pyHalo.Spatial.nfw import NFW3D
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Rendering.keywords import subhalo_mass_function
from pyHalo.Spatial.keywords import subhalo_spatial_NFW

class MainLensPowerLaw(object):

    def __init__(self, args, geometry, x_center_lens, y_center_lens):

        self._geometry = geometry

        zlens, zsource = geometry._zlens, geometry._zsource
        kpc_per_arcsec_zlens = geometry._kpc_per_arcsec_zlens

        lenscosmo = LensCosmo(zlens, zsource, geometry._cosmo)

        parameterization_args = subhalo_mass_function(args, kpc_per_arcsec_zlens, zlens)
        spatial_args = subhalo_spatial_NFW(args, kpc_per_arcsec_zlens, zlens, lenscosmo)

        self._mass_func_parameterization = BrokenPowerLaw(**parameterization_args)

        self._spatial_parameterization = NFW3D(**spatial_args)

        self._center_x, self._center_y = x_center_lens, y_center_lens

    def __call__(self):
        """

        :return: x coordinate, y coordinates, r3d, r3d
        NOTE: x and y are returned in arcsec, while r2d and r3d are expressed in kpc
        """
        masses = self._mass_func_parameterization.draw()

        if len(masses) > 0:

            # EVERYTHING EXPRESSED IN KPC
            x_kpc, y_kpc, r2d_kpc, r3d_kpc = self._spatial_parameterization.draw(len(masses))

            x_arcsec = np.array(x_kpc) * self._geometry._kpc_per_arcsec_zlens ** -1 + self._center_x
            y_arcsec = np.array(y_kpc) * self._geometry._kpc_per_arcsec_zlens ** -1 + self._center_y

            return np.array(masses), np.array(x_arcsec), np.array(y_arcsec), np.array(r2d_kpc), np.array(r3d_kpc), np.array(
                [self._geometry._zlens] * len(masses))
        else:
            return np.array(masses), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

