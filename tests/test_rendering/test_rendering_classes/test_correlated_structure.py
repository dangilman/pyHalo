from pyHalo.single_realization import SingleHalo
from pyHalo.Rendering.correlated_structure import CorrelatedStructure
import numpy as np
from scipy.interpolate import interp1d
import numpy.testing as npt
import pytest
from pyHalo.Cosmology.cosmology import Cosmology

class TestCorrelated(object):

    def setup(self):

        realization = SingleHalo(10 ** 8, 0., -0.1, 'TNFW', 0.02, 0.5, 1.01, subhalo_flag=False)
        zlist = np.arange(0.02, 0.98, 0.02)
        rmax = 0.3
        for i, zi in enumerate(zlist):
            theta = np.random.uniform(0., 2 * np.pi)
            r = np.random.uniform(0, rmax ** 2) ** 0.5
            xi, yi = np.cos(theta) * r, np.sin(theta) * r
            mi = np.random.uniform(7, 8)
            single_halo = SingleHalo(10 ** mi, xi, yi, 'TNFW', zi, 0.5, 1.01, subhalo_flag=False)
            realization = realization.join(single_halo)

        cosmo = Cosmology()
        self.realization = realization
        self.rmax = rmax

        zlist = np.arange(0.00, 1.02, 0.02)
        x_image = [0.] * len(zlist)
        y_image = [0.] * len(zlist)
        dlist = [cosmo.D_C_transverse(zi) for zi in zlist]

        self.x_image_interp_list = [interp1d(dlist, x_image)]
        self.y_image_interp_list = [interp1d(dlist, y_image)]

    def test_delta_function(self):

        kwargs_rendering = {'mass_function_type': 'DELTA', 'logM': 5., 'mass_fraction': 0.5}
        correlated = CorrelatedStructure(kwargs_rendering, self.realization, self.rmax)
        masses, x, y, r3d, redshifts, subhalo_flag = correlated.render(self.x_image_interp_list,
                                                                       self.y_image_interp_list,
                                                                       0.002)

        npt.assert_equal(len(masses), len(x))
        npt.assert_equal(len(masses), len(y))

        for i in range(0, len(masses)):
            npt.assert_equal(np.log10(masses), 5.)
            npt.assert_equal(np.hypot(x[i], y[i]) <= self.rmax * np.sqrt(2), True)
            npt.assert_equal(True, r3d[i] is None)

if __name__ == '__main__':
   pytest.main()
