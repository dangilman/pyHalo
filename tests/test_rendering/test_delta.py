import numpy.testing as npt
import pytest
import numpy as np
from pyHalo.Rendering.MassFunctions.delta import DeltaFunction

class TestBackgroundDensityDelta(object):

    def setup(self):

        self.mass = 0.01
        self.volume = 10
        self.rho = 10
        self.mfunc = DeltaFunction(self.mass, self.volume, self.rho, False)
        self.mfunc_poisson = DeltaFunction(self.mass, self.volume, self.rho, True)
        self.mfunc_empty = DeltaFunction(100000 * self.volume * self.rho, self.volume, self.rho, False)

    def test_density_delta(self):

        n_expected = self.rho * self.volume / self.mass
        m = self.mfunc.draw()
        n_drawn = len(m)
        npt.assert_equal(n_drawn, n_expected)
        for mi in m:
            npt.assert_equal(mi, self.mass)

        m = self.mfunc_poisson.draw()
        for mi in m:
            npt.assert_equal(mi, self.mass)

        m = self.mfunc_empty.draw()
        npt.assert_equal(len(m), 0.)

if __name__ == '__main__':

    pytest.main()
