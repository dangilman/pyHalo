import pytest
import numpy.testing as npt
from pyHalo.pyhalo import pyHalo
from pyHalo.defaults import lenscone_default

class TestpyHaloBase(object):

    def setup(self):

        self.pyhalo = pyHalo(0.5, 2.)

    def test_lens_plane_redshifts(self):

        zplanes, dz = self.pyhalo.lens_plane_redshifts
        npt.assert_equal(len(zplanes), len(dz))
        npt.assert_equal(dz[1], lenscone_default.default_z_step)
        npt.assert_equal(True, zplanes[-1]==2 - lenscone_default.default_z_step)

if __name__ == '__main__':

    pytest.main()
