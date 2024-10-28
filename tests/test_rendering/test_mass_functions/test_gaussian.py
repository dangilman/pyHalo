import pytest
import numpy as np
import numpy.testing as npt
from pyHalo.Rendering.MassFunctions.gaussian import Gaussian

class TestGaussian(object):

    def setup_method(self):
        n0 = 5000
        mean = 5.0
        sigma = 0.5
        mfunc = Gaussian(n0, mean, sigma, draw_poisson=False)
        mfunc_piosson = Gaussian(n0, mean, sigma, draw_poisson=True)
        self.mean = mean
        self.sigma = sigma
        self.n0 = n0
        self.mass_function_poisson = mfunc_piosson
        self.mfunc = mfunc

    def test_mass_function(self):

        m = self.mass_function_poisson.draw()
        log10_m = np.log10(m)
        mu, sigma = np.mean(log10_m), np.std(log10_m)
        npt.assert_almost_equal(mu, self.mean, 2)
        npt.assert_almost_equal(sigma, self.sigma, 2)
        npt.assert_almost_equal(mu, self.mass_function_poisson.first_moment,2)

    def test_draw_poisson(self):

        m = self.mfunc.draw()
        m_poisson = self.mass_function_poisson.draw()
        npt.assert_equal(True, len(m)!=len(m_poisson))

if __name__ == '__main__':
   pytest.main()
