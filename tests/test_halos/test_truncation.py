import pytest
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Cosmology.cosmology import Cosmology
import numpy.testing as npt
from pyHalo.Halos.tidal_truncation import TruncationRN, TruncationRoche, TruncationSplashBack, TruncateMeanDensity
from pyHalo.truncation_models import truncation_models
from astropy.cosmology import FlatLambdaCDM
from pyHalo.Halos.concentration import ConcentrationDiemerJoyce
import os


class TestTruncation(object):

    def setup_method(self):

        astropy = FlatLambdaCDM(70.0, 0.3)
        cosmo = Cosmology(astropy)
        self.lenscosmo = LensCosmo(0.5, 1.5, cosmo)

    def test_load_models(self):

        model_name_list = ['TRUNCATION_R50', 'TRUNCATION_RN', 'TRUNCATION_ROCHE', 'TRUNCATION_ROCHE_GILMAN2020',
                           'SPLASHBACK', 'TRUNCATION_MEAN_DENSITY']
        kwargs_model_list = [{}, {'LOS_truncation_factor': 50.}, {'RocheNorm': 1.0, 'm_power': 1./3, 'RocheNu': 2.0/3.0}, {},
                             {}, {}]
        for model,kwargs in zip(model_name_list, kwargs_model_list):
            mod, kw = truncation_models(model)
            kwargs.update(kw)
            kwargs['lens_cosmo'] = self.lenscosmo
            _ = mod(**kwargs)

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

    def test_truncation_splashback(self):

        class DummyHalo(object):
            def __init__(self, m, z):
                self.mass = m
                self.z = z
        truncation_splashback = TruncationSplashBack(self.lenscosmo)
        rt = truncation_splashback.truncation_radius(10**8, 0.4)
        halo = DummyHalo(10**8, 0.4)
        rt_halo = truncation_splashback.truncation_radius_halo(halo)
        npt.assert_almost_equal(rt, rt_halo)

    def test_truncation_mean_density(self):

        class DummyHalo(object):
            def __init__(self, m, z):
                self.mass = m
                self.z = z
                self.z_eval = z
                self.rperi_units_r200 = 0.7
                self.c = 16.0

        halo = DummyHalo(10 ** 8, 0.4)
        median_rt_over_rs = 2.0
        c_power = 4.0
        cmodel = ConcentrationDiemerJoyce(self.lenscosmo.cosmo, scatter=False)
        c_theory = cmodel.nfw_concentration(10 ** 8, 0.4)
        truncation = TruncateMeanDensity(self.lenscosmo, median_rt_over_rs, c_power)
        r_t = truncation.truncation_radius(10**8, 0.4, c_theory, 16.0, 0.7)
        r_t_halo = truncation.truncation_radius_halo(halo)
        npt.assert_almost_equal(r_t, r_t_halo)

        rt_over_rs_theory = median_rt_over_rs * (16.0 / c_theory) ** c_power * (0.7 / 0.5)
        rs = self.lenscosmo.NFW_params_physical(10**8, 16.0, 0.4)[1]
        npt.assert_almost_equal(rs * rt_over_rs_theory, r_t)


if __name__ == '__main__':
    pytest.main()
