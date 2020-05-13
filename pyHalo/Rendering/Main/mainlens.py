import numpy as np

from pyHalo.Rendering.Main.base import MainLensBase

class MainLensPowerLaw(MainLensBase):

    def __call__(self):
        """

        :return: x coordinate, y coordinates, r3d, r3d
        NOTE: x and y are returned in arcsec, while r2d and r3d are expressed in kpc
        """
        masses = self._mass_func_parameterization.draw()

        if len(masses) > 0:

            # EVERYTHING EXPRESSED IN KPC
            x_kpc, y_kpc, r2d_kpc, r3d_kpc = self.spatial_parameterization.draw(len(masses),
                                                                                self.geometry._zlens)

            x_arcsec = np.array(x_kpc) * self.geometry._kpc_per_arcsec_zlens ** -1 + self._center_x
            y_arcsec = np.array(y_kpc) * self.geometry._kpc_per_arcsec_zlens ** -1 + self._center_y

            return np.array(masses), np.array(x_arcsec), np.array(y_arcsec), np.array(r2d_kpc), np.array(r3d_kpc), np.array(
                [self.geometry._zlens] * len(masses))
        else:
            return np.array(masses), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
