import pytest
import numpy as np
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Cosmology.cosmology import Cosmology
import numpy.testing as npt
from pyHalo.Halos.tidal_truncation import TruncationRN, TruncationRoche
from astropy.cosmology import FlatLambdaCDM

class TestTruncation(object):

    def setup_method(self):

        astropy = FlatLambdaCDM(70.0, 0.3)
        cosmo = Cosmology(astropy)
        self.lenscosmo = LensCosmo(0.5, 1.5, cosmo)

    def test_truncation_RN(self):

        N = 200
        halo_mass = 10 ** 8
        truncation_RN = TruncationRN(self.lenscosmo, N)
        r200_kpc = truncation_RN.truncation_radius(halo_mass, 0.5)
        r200_kpc_true = self.lenscosmo.NFW_params_physical(halo_mass, 16.0, 0.5)[-1]
        npt.assert_almost_equal(r200_kpc, r200_kpc_true)

    def test_truncation_roche(self):

        norm = 1.4
        m_power = 1. / 3
        nu = 2. / 3
        r3d_subhalo = 65.0
        halo_mass = 10 ** 8
        truncation_roche = TruncationRoche(None, norm, m_power, nu)
        r200_kpc = truncation_roche.truncation_radius(halo_mass, r3d_subhalo)
        r200_kpc_true = norm * (halo_mass/10**7) ** m_power * (r3d_subhalo/50)**nu
        npt.assert_almost_equal(r200_kpc, r200_kpc_true, 3)

if __name__ == '__main__':
    pytest.main()
