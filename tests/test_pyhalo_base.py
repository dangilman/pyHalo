import pytest
import numpy.testing as npt
import numpy as np
from pyHalo.pyhalo import pyHalo
from lenstronomy.LensModel.lens_model import LensModel

class TestpyHaloBase(object):

    def setup(self):

        self.pyhalo = pyHalo(0.5, 2.)

    def test_interp_ray_paths(self):

        x = [1.4, -1.]
        y = [-0.4, 0.2]
        lens_model = LensModel(['SIS'], z_source=2., lens_redshift_list=[0.5], multi_plane=True)
        kwargs_lens = [{'theta_E': 1., 'center_x': 0., 'center_y': 0.}]

        interpx, interpy = self.pyhalo.interpolate_ray_paths(x, y, lens_model, kwargs_lens, 2.,
                               terminate_at_source=False, source_x=None, source_y=None, evaluate_at_mean=False)
        zarray = np.linspace(0., 2., 100)
        dc = [self.pyhalo.cosmology.D_C_transverse(zi) for zi in zarray]
        interpx_mean, interpy_mean = self.pyhalo.interpolate_ray_paths(x, y, lens_model, kwargs_lens, 2.,
                                                             terminate_at_source=False, source_x=None, source_y=None,
                                                             evaluate_at_mean=True)
        for dci in dc:
            x, y = 0.5*(interpx[0](dci) + interpx[1](dci)), 0.5*(interpy[0](dci) + interpy[1](dci))
            npt.assert_almost_equal(x, interpx_mean[0](dci))
            npt.assert_almost_equal(y, interpy_mean[0](dci))

        x = [1.4, -1.]
        y = [-0.4, 0.2]
        source_x, source_y = 0.2, 0.5
        interpx, interpy = self.pyhalo.interpolate_ray_paths(x, y, lens_model, kwargs_lens, 2.,
                                                                       terminate_at_source=True, source_x=source_x,
                                                                       source_y=source_y,
                                                                       evaluate_at_mean=True)
        npt.assert_almost_equal(interpx[0](dc[-1]), source_x)
        npt.assert_almost_equal(interpy[0](dc[-1]), source_y)

if __name__ == '__main__':

    pytest.main()
