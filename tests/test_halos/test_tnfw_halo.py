import numpy.testing as npt
from lenstronomy.Cosmo.lens_cosmo import LensCosmo as LensCosmoLenstronomy
from pyHalo.Halos.HaloModels.TNFW import TNFWSubhalo, TNFWFieldHalo
from pyHalo.Halos.concentration import ConcentrationDiemerJoyce
from pyHalo.Halos.tidal_truncation import TruncationRoche
from astropy.cosmology import FlatLambdaCDM
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Cosmology.cosmology import Cosmology
import pytest
import numpy as np

class TestTNFWHalos(object):

    def setup_method(self):

        astropy = FlatLambdaCDM(70.0, 0.3)
        cosmo = Cosmology(astropy_instance=astropy)
        self.zhalo = 0.5
        self.zsource = 2.0
        self.lens_cosmo = LensCosmo(self.zhalo, self.zsource, cosmo)
        self.truncation_class = TruncationRoche(None, 100000000.0)
        self.concentration_class = ConcentrationDiemerJoyce(self.lens_cosmo, scatter=False)
        self.lclenstronomy = LensCosmoLenstronomy(self.zhalo, self.zsource, astropy)

    def test_simple_setup(self):
        mass = 10 ** 8
        x = 0.0
        y = 0.0
        z = 0.5
        tau = 2.2
        halo = TNFWFieldHalo.simple_setup(mass, x, y, z, tau, self.lens_cosmo)
        kwargs, _ = halo.lenstronomy_params
        npt.assert_almost_equal(kwargs[0]['r_trunc']/kwargs[0]['Rs'], tau, 3)

    def test_lenstronomy_params(self):

        m = 10 ** 8
        x = 0.5
        y = 1.0
        r3d = 100
        unique_tag = 1.0
        is_subhalo = False
        nfw_field_halo = TNFWFieldHalo(m, x, y, r3d, self.zhalo, is_subhalo, self.lens_cosmo, {},
                                      self.truncation_class, self.concentration_class, unique_tag)

        kwargs_halo, _ = nfw_field_halo.lenstronomy_params
        id = nfw_field_halo.lenstronomy_ID
        npt.assert_string_equal('TNFW', id[0])

        rho0, Rs, c, r200, M200 = self.lclenstronomy.nfw_angle2physical(kwargs_halo[0]['Rs'],
                                                                        kwargs_halo[0]['alpha_Rs'])
        npt.assert_almost_equal(M200/m, 1.0, 3)

        is_subhalo = True
        kwargs_profile = {'evaluate_mc_at_zlens': False}
        nfw_subhalo = TNFWSubhalo(m, x, y, r3d, self.zhalo, is_subhalo, self.lens_cosmo, kwargs_profile,
                                  self.truncation_class, self.concentration_class, unique_tag)
        c_subhalo = nfw_subhalo.c
        c_field = nfw_field_halo.c
        npt.assert_equal(True, c_subhalo <= c_field)

    def test_z_infall(self):
        m = 10 ** 8
        x = 0.5
        y = 1.0
        r3d = 100
        unique_tag = 1.0
        kwargs_profile = {'evaluate_mc_at_zlens': False, 'c_scatter': False, 'c_scatter_dex': 0.2}
        is_subhalo = True
        tnfw_subhalo = TNFWSubhalo(m, x, y, r3d, self.zhalo, is_subhalo, self.lens_cosmo, kwargs_profile,
                                  self.truncation_class, self.concentration_class, unique_tag)
        z_infall = tnfw_subhalo.z_infall
        npt.assert_equal(True, self.zhalo <= z_infall)

    def test_bound_mass(self):
        m = 10 ** 8
        x = 0.5
        y = 1.0
        r3d = 100
        unique_tag = 1.0
        kwargs_profile = {'evaluate_mc_at_zlens': False, 'c_scatter': False, 'c_scatter_dex': 0.2}
        is_subhalo = True
        truncation_class = TruncationRoche(None, 2.0)
        tnfw_subhalo = TNFWSubhalo(m, x, y, r3d, self.zhalo, is_subhalo, self.lens_cosmo, kwargs_profile,
                                  truncation_class, self.concentration_class, unique_tag)
        bound_mass = tnfw_subhalo.bound_mass
        npt.assert_equal(True, bound_mass/10**8 < 1)

    def test_density_mass(self):

        m = 10 ** 8
        x = 0.5
        y = 1.0
        r3d = 100
        unique_tag = 1.0
        kwargs_profile = {'evaluate_mc_at_zlens': False, 'c_scatter': False, 'c_scatter_dex': 0.2}
        is_subhalo = False
        tnfw_fieldhalo = TNFWFieldHalo(m, x, y, r3d, self.zhalo, is_subhalo, self.lens_cosmo, kwargs_profile,
                                  self.truncation_class, self.concentration_class, unique_tag)
        rho_s, rs, r200 = tnfw_fieldhalo.nfw_params
        c = tnfw_fieldhalo.c
        mtheory = 4 * np.pi * rho_s * rs ** 3 * (np.log(1 + c) - c/(1+c))
        npt.assert_almost_equal(mtheory, tnfw_fieldhalo.mass)
        rmax = c * rs
        m_calculated = tnfw_fieldhalo.mass_3d(rmax)
        npt.assert_almost_equal(mtheory/m_calculated, 1.0)

        tau = 2.
        tnfw_halo = TNFWFieldHalo.simple_setup(m, 0.0, 0.0, 0.6, tau, self.lens_cosmo)
        rs = tnfw_halo.nfw_params[1]
        r = np.linspace(0.001, tnfw_halo.c, 50000) * rs
        rho = tnfw_halo.density_profile_3d_lenstronomy(r)
        m_exact = np.trapz(4 * np.pi * rho * r ** 2, r)
        m_class = tnfw_halo.mass_3d('r200')
        npt.assert_almost_equal(m_class/ m_exact,1,3)

if __name__ == '__main__':
    pytest.main()

