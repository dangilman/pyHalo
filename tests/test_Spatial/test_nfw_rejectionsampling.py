import numpy as np
import numpy.testing as npt
import pytest
from pyHalo.Spatial.nfw_core import NFW3DCoreRejectionSampling
from lenstronomy.LensModel.Profiles.cnfw import CNFW

class TestNFWRejSamp(object):

    def setup(self):

        prof = CNFW()
        self.func = prof._F
        self.rs = 60
        self.rmax2d = 40
        self.rvir = 350
        self.rcore = 10.
        self.nfw = NFW3DCoreRejectionSampling(self.rs, self.rmax2d, self.rvir, self.rcore)

    def test_limit(self):

        x, y, r3 = self.nfw.draw(2000, None)
        r2 = np.hypot(x, y)
        npt.assert_almost_equal(max(r2)/self.rmax2d, 1, 2)

    def test_profile(self):

        x, y, r3 = self.nfw.draw(25000, None)
        r2 = np.hypot(x, y)
        rbins = np.arange(10, self.rmax2d+5, 5)
        n = []
        for i in range(0, len(rbins) - 1):
            condition = np.logical_and(r2 >= rbins[i], r2 < rbins[i + 1])
            N = np.sum(condition)
            area = np.pi * (rbins[i + 1] ** 2 - rbins[i] ** 2)
            n.append(N / area)

        x = rbins / self.rs
        xtidal = self.rcore/self.rs
        true = self.func(x, xtidal)
        true *= max(true) ** -1
        n = np.array(n)/max(n)

        for i in range(0, len(true)-1):
            npt.assert_almost_equal(true[i]/n[i], 1, 1)

if __name__ == '__main__':

    pytest.main()
