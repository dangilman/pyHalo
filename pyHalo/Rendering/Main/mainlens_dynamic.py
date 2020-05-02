import numpy as np
from pyHalo.Rendering.Main.mainlens import MainLensPowerLaw

class MainLensPowerLawDynamic(object):

    def __init__(self, args, geometry, x_center_lens, y_center_lens):

        self.main = MainLensPowerLaw(args, geometry, x_center_lens, y_center_lens)

        self._masses, self._x_arcsec, self._y_arcsec, self._r2d_kpc, self._r3d_kpc, _ = self.main()

    def __call__(self, center_x, center_y, log_mlow, log_mhigh, aperture_radius):
        """

        :return: x coordinate, y coordinates, r3d, r3d
        NOTE: x and y are returned in arcsec, while r2d and r3d are expressed in kpc
        """

        if len(self._masses) > 0:
            dx = self._x_arcsec - center_x
            dy = self._y_arcsec - center_y
            dr = np.sqrt(dx ** 2 + dy ** 2)

            mlow, mhigh = 10**log_mlow, 10**log_mhigh

            cond1 = self._masses >= mlow
            cond2 = self._masses < mhigh
            cond3 = dr <= aperture_radius

            inds = np.where(cond1 & cond2 & cond3)

            redshifts = [self.main._geometry._zlens] * len(dx[inds])

            return self._masses[inds], self._x_arcsec[inds], self._y_arcsec[inds], self._r2d_kpc[inds], \
                   self._r3d_kpc[inds], redshifts

        else:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
