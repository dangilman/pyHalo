import numpy.testing as npt
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Halos.lens_cosmo import LensCosmo
import numpy as np
import pytest
from scipy.integrate import quad

class TestLensCosmo(object):

    def setup(self):

        self.cosmo = Cosmology()
        zlens, zsource = 0.5, 1.5
        self.lens_cosmo = LensCosmo(zlens, zsource, self.cosmo)

    def test_nfw_params(self):

        for z in [0., 0.74]:

            M, c = 10**8, 17.5
            a_z = (1 + z) ** -1
            rho_crit_0 = self.cosmo.rho_crit(0.)

            rhos, rs, r200 = self.lens_cosmo._nfwParam_physical_Mpc(M, c, z)

            def _profile(x):
                fac = x * (1 + x) ** 2
                return 1. / fac
            def _integrand(x):
                return 4 * np.pi * x ** 2 * _profile(x)

            volume = 4*np.pi/3 * r200 ** 3
            mean_density_physical = rhos * rs ** 3 * quad(_integrand, 0, r200/rs)[0] / volume
            mean_density_comoving = mean_density_physical * a_z ** 3
            ratio = mean_density_comoving/rho_crit_0

            npt.assert_almost_equal(ratio, 200)

if __name__ == '__main__':
    pytest.main()


