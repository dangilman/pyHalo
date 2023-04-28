from pyHalo.Rendering.SpatialDistributions.uniform import Uniform, LensConeUniform
from pyHalo.Cosmology.geometry import Geometry
from pyHalo.Cosmology.cosmology import Cosmology
import numpy as np
import numpy.testing as npt
import pytest

class TestUniform(object):

    def setup_method(self):

        cosmo = Cosmology()
        self.geometry = Geometry(cosmo, 0.5, 1.5, 6., 'DOUBLE_CONE')
        self.uni = Uniform(3, self.geometry)
        self.lenscone_uni = LensConeUniform(6., self.geometry)

    def test_uniform(self):

        x, y = self.uni.draw(1000, 0.3, 1., 0., 0.)
        kpc_per_asec = self.geometry.kpc_per_arcsec(0.3)
        x *= kpc_per_asec ** -1
        y *= kpc_per_asec ** -1
        r2d = np.hypot(x, y)
        npt.assert_equal(True, max(r2d) < 3)

        x, y = self.uni.draw(1000, 0.3, 0.5, 0., 0.)
        x *= kpc_per_asec ** -1
        y *= kpc_per_asec ** -1
        r2d = np.hypot(x, y)
        npt.assert_equal(True, max(r2d) < 1.5)

    def test_lens_cone_uniform(self):

        x, y = self.lenscone_uni.draw(10000, 0.5)
        kpc_per_asec = self.geometry.kpc_per_arcsec(0.5)
        x *= kpc_per_asec ** -1
        y *= kpc_per_asec ** -1
        x2, y2 = self.lenscone_uni.draw(10000, 0.9)
        kpc_per_asec = self.geometry.kpc_per_arcsec(0.9)
        x2 *= kpc_per_asec ** -1
        y2 *= kpc_per_asec ** -1

        scale1 = self.geometry.rendering_scale(0.5)
        scale2 = self.geometry.rendering_scale(0.9)

        max1 = max(np.hypot(x, y))
        max2 = max(np.hypot(x2, y2))
        npt.assert_almost_equal(max1/max2, scale1/scale2, 2)

    def test_distribution(self):

        x, y = self.uni.draw(1000000, 0.2, 1., 0., 0.)
        r2 = np.hypot(x, y)
        rbins = np.arange(1., 3.+0.5, 0.5)
        n = []
        for i in range(0, len(rbins)-1):
            condition = np.logical_and(r2 >= rbins[i], r2<rbins[i+1])
            N = np.sum(condition)
            area = np.pi * (rbins[i+1]**2 - rbins[i]**2)
            n.append(N/area)
        npt.assert_almost_equal(n[0]/n[1], 1, 1)
        npt.assert_almost_equal(n[2]/n[3], 1, 1)

if __name__ == '__main__':

    pytest.main()


