import pytest
import numpy.testing as npt
from pyHalo.Halos.tidal_truncation import TruncationGalacticus, TruncationGalacticusKeeley24
from pyHalo.Halos.HaloModels.TNFW import TNFWSubhalo
from pyHalo.Halos.concentration import ConcentrationDiemerJoyce
import numpy as np
from pyHalo.Halos.lens_cosmo import LensCosmo

class TestTruncationGalacticus(object):

    def setup_method(self):

        self.lens_cosmo_instance = LensCosmo(0.5, 1.5)
        chost = 5.0
        self.tg = TruncationGalacticus(self.lens_cosmo_instance, chost)
        self.tgk24 = TruncationGalacticusKeeley24(self.lens_cosmo_instance, chost)

    def test_interp_k24(self):

        interp = self.tgk24._mass_loss_interp

        log10c = 0.5
        m1 = interp.evaluate_mean_mass_loss(log10c, 1.0, 5.0)
        log10c = 2.0
        m2 = interp.evaluate_mean_mass_loss(log10c, 1.0, 5.0)
        npt.assert_equal(m2 > m1, True)

        m = interp.evaluate_mean_mass_loss(np.array([0.5, 2.0]), 1.0, 5.0)
        npt.assert_equal(m[0], m1)
        npt.assert_equal(m[1], m2)

        log10c = 0.5
        m1 = interp.evaluate_scatter_dex(log10c, 1.0, 5.0)
        log10c = 2.0
        m2 = interp.evaluate_scatter_dex(log10c, 1.0, 5.0)
        npt.assert_equal(m1 > m2, True)

        m = interp.evaluate_scatter_dex(np.array([0.5, 2.0]), 1.0, 5.0)
        npt.assert_equal(m[0], m1)
        npt.assert_equal(m[1], m2)

    def test_evaluate_keeley24(self):

        halo_mass = 10**8
        infall_concentration = 10.0
        time_since_infall = 2.0
        chost = 5.0
        z_eval_angles = 0.5
        rt_kpc = self.tgk24.truncation_radius(halo_mass, infall_concentration, time_since_infall,
                                        chost, z_eval_angles)
        npt.assert_equal(True, np.isfinite(rt_kpc))
        npt.assert_equal(True, isinstance(rt_kpc, float))

        halo_mass = 10 ** 8
        infall_concentration = 10.0
        time_since_infall = 8.0
        chost = 5.0
        z_eval_angles = 0.5
        rt_kpc = self.tgk24.truncation_radius(halo_mass, infall_concentration, time_since_infall,
                                            chost, z_eval_angles)

        npt.assert_equal(True, np.isfinite(rt_kpc))
        npt.assert_equal(True, isinstance(rt_kpc, float))

        class DummyHalo(object):

            def __init__(self):
                self.mass = 10**8
                self.c = 16.0
                self.time_since_infall = 5.0
                self.z = 0.5
                self.z_eval = 0.5

        halo = DummyHalo()
        rt_kpc = self.tgk24.truncation_radius_halo(halo)
        npt.assert_equal(True, np.isfinite(rt_kpc))
        npt.assert_equal(True, isinstance(rt_kpc, float))

    def test_evaluate(self):

        halo_mass = 10**8
        infall_concentration = 10.0
        time_since_infall = 2.0
        z_eval_angles = 0.5
        rt_kpc = self.tg.truncation_radius(halo_mass, infall_concentration, time_since_infall,
                                        z_eval_angles)
        npt.assert_equal(True, np.isfinite(rt_kpc))
        npt.assert_equal(True, isinstance(rt_kpc, float))

        halo_mass = 10 ** 8
        infall_concentration = 10.0
        time_since_infall = 8.0
        z_eval_angles = 0.5
        rt_kpc = self.tg.truncation_radius(halo_mass, infall_concentration, time_since_infall,
                                            z_eval_angles)

        npt.assert_equal(True, np.isfinite(rt_kpc))
        npt.assert_equal(True, isinstance(rt_kpc, float))

        class DummyHalo(object):

            def __init__(self):
                self.mass = 10**8
                self.c = 16.0
                self.time_since_infall = 5.0
                self.z = 0.5
            def rescale_normalization(self, rescale):
                return

        halo = DummyHalo()
        rt_kpc = self.tg.truncation_radius_halo(halo)
        npt.assert_equal(True, np.isfinite(rt_kpc))
        npt.assert_equal(True, isinstance(rt_kpc, float))

    def test_tg(self):
        lens_cosmo = LensCosmo(0.5, 2.0,None)
        truncation_class = TruncationGalacticus(lens_cosmo, c_host=4.0)
        concentration_class = ConcentrationDiemerJoyce(lens_cosmo.cosmo, scatter=False)
        mass = 10 ** 8
        x = 0
        y = 0
        r3d = None
        z = 0.5
        sub_flag = True
        halo_args = {}
        unique_tag = 1
        tnfw_profile_1 = TNFWSubhalo(mass, x, y, r3d, z, sub_flag,
                                     lens_cosmo, halo_args, truncation_class, concentration_class, unique_tag)
        tnfw_profile_1._c = 3.0
        tnfw_profile_1._time_since_infall = 6.0

        m_bound = tnfw_profile_1.bound_mass
        rescale_norm = tnfw_profile_1._rescale_norm
        kwargs, _ = tnfw_profile_1.lenstronomy_params
        npt.assert_equal(rescale_norm<1, True)
        npt.assert_equal(m_bound < mass, True)

if __name__ == '__main__':
    pytest.main()
