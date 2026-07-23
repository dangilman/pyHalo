import numpy.testing as npt
import numpy as np
import pytest
import inspect
import re
from astropy.cosmology import FlatLambdaCDM
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Halos.concentration import ConcentrationDiemerJoyce, ConcentrationConstant
from pyHalo.Halos.tidal_truncation import (TruncationGalacticus, Multiple_RS,
                                           TruncationRN, ConstantTruncationArcsec)
from pyHalo.Halos.concentration import ConcentrationWDMPolynomial
from pyHalo.Halos.HaloModels.TNFW import TNFWSubhalo, TNFWFieldHalo
from pyHalo.Halos.HaloModels.NFW import NFWSubhhalo, NFWFieldHalo
from pyHalo.Halos.HaloModels.globular_cluster import GlobularCluster
from pyHalo.Halos.HaloModels.NFW_core_trunc import TNFWCHaloEvolving
from pyHalo.Halos.galacticus_truncation.transfer_function_density_profile import compute_r_te_and_f_t
from pyHalo.Halos.batch_halo_util import (nfw_params_physical_vectorized,
                                          batch_nfw_concentration,
                                          precompute_tnfw_field_halos,
                                          compute_r_te_and_f_t_vectorized,
                                          precompute_concentrations,
                                          precompute_infall_times,
                                          precompute_tnfw_subhalos,
                                          precompute_tnfw_bound_masses,
                                          precompute_sidm_evolving_profiles,
                                          precompute_realization,
                                          precompute_nfw_params)
from pyHalo.single_realization import Realization


class TestBatchUtil(object):

    def setup_method(self):

        astropy = FlatLambdaCDM(70.0, 0.3)
        cosmo = Cosmology(astropy_instance=astropy)
        self.zlens = 0.5
        self.zsource = 2.0
        self.lens_cosmo = LensCosmo(self.zlens, self.zsource, cosmo)
        self.concentration_class = ConcentrationDiemerJoyce(self.lens_cosmo.cosmo.astropy, scatter=False)
        self.truncation_class = TruncationGalacticus(self.lens_cosmo, c_host=6.0)

    def _make_subhalos(self, n, seed=42, mdef='TNFW'):

        np.random.seed(seed)
        halos = []
        for _ in range(n):
            m = 10 ** np.random.uniform(6, 10.7)
            x, y = np.random.uniform(-2, 2, 2)
            if mdef == 'TNFW':
                halos.append(TNFWSubhalo(m, x, y, None, self.zlens, True, self.lens_cosmo,
                                         {}, self.truncation_class, self.concentration_class,
                                         np.random.rand()))
            else:
                halos.append(NFWSubhhalo(m, x, y, None, self.zlens, True, self.lens_cosmo,
                                         {}, None, self.concentration_class,
                                         np.random.rand()))
        return halos

    def _make_fieldhalos(self, n, seed=42, mdef='TNFW'):

        np.random.seed(seed)
        halos = []
        for _ in range(n):
            m = 10 ** np.random.uniform(6, 10.7)
            x, y = np.random.uniform(-2, 2, 2)
            if mdef == 'TNFW':
                halos.append(TNFWFieldHalo(m, x, y, None, self.zlens, False, self.lens_cosmo,
                                           {}, self.truncation_class, self.concentration_class,
                                           np.random.rand()))
            else:
                halos.append(NFWFieldHalo(m, x, y, None, self.zlens, False, self.lens_cosmo,
                                          {}, None, self.concentration_class,
                                          np.random.rand()))
        return halos

    def _make_globular_clusters(self, n, seed=42):

        np.random.seed(seed)
        clusters = []
        kwargs_gc = {'gamma': 6.0, 'gc_size_pc': 100.0, 'gc_concentration': 20.0}
        for _ in range(n):
            m = 10 ** np.random.uniform(4, 5)
            x, y = np.random.uniform(-2, 2, 2)
            clusters.append(GlobularCluster(m, x, y, self.zlens, self.lens_cosmo,
                                            dict(kwargs_gc), np.random.rand()))
        return clusters

    def test_nfw_params_physical_vectorized(self):

        np.random.seed(0)
        n = 100
        m = 10 ** np.random.uniform(6, 11, n)
        c = np.random.uniform(2, 30, n)
        z = np.random.uniform(0.1, 4, n)
        for pseudo_nfw in [False, True]:
            ref = np.array([self.lens_cosmo.NFW_params_physical(mi, ci, zi, pseudo_nfw)
                            for (mi, ci, zi) in zip(m, c, z)])
            rhos, rs, r200 = nfw_params_physical_vectorized(self.lens_cosmo, m, c, z, pseudo_nfw)
            npt.assert_almost_equal(rhos / ref[:, 0], 1, 6)
            npt.assert_almost_equal(rs / ref[:, 1], 1, 6)
            npt.assert_almost_equal(r200 / ref[:, 2], 1, 6)

    def test_compute_r_te_and_f_t_vectorized(self):

        np.random.seed(0)
        n = 100
        m_infall = 10 ** np.random.uniform(7, 10, n)
        c_infall = np.random.uniform(3, 30, n)
        m_bound = 10 ** np.random.uniform(-3, -0.01, n) * m_infall
        _, _, r200 = nfw_params_physical_vectorized(self.lens_cosmo, m_infall, c_infall,
                                                    np.full(n, self.zlens))
        rte_vec, ft_vec = compute_r_te_and_f_t_vectorized(m_bound, m_infall, r200, c_infall)
        for i in range(n):
            rte, ft = compute_r_te_and_f_t(m_bound[i], m_infall[i], r200[i], c_infall[i])
            npt.assert_almost_equal(rte_vec[i] / rte, 1, 6)
            npt.assert_almost_equal(ft_vec[i] / ft, 1, 6)

    def test_precompute_concentrations(self):

        halos_batch = self._make_subhalos(50) + self._make_fieldhalos(50)
        halos_reference = self._make_subhalos(50) + self._make_fieldhalos(50)
        z_infall = np.random.uniform(self.zlens + 0.01, 4.0, 50)
        for i in range(50):
            halos_batch[i]._z_infall = float(z_infall[i])
            halos_reference[i]._z_infall = float(z_infall[i])
        precompute_concentrations(halos_batch)
        for hb, hr in zip(halos_batch, halos_reference):
            npt.assert_almost_equal(hb.c / hr.c, 1, 6)

    def test_precompute_infall_times(self):

        halos_batch = self._make_subhalos(50)
        halos_reference = self._make_subhalos(50)
        z_infall = np.random.uniform(self.zlens + 0.01, 4.0, 50)
        for i in range(50):
            halos_batch[i]._z_infall = float(z_infall[i])
            halos_reference[i]._z_infall = float(z_infall[i])
        precompute_infall_times(halos_batch, self.lens_cosmo)
        for hb, hr in zip(halos_batch, halos_reference):
            npt.assert_almost_equal(hb._time_since_infall, hr.time_since_infall, 6)

    def test_precompute_tnfw_subhalos(self):

        # pin the random stages (z_infall, bound mass) on both sides so the
        # remainder of the pipeline can be compared deterministically
        halos_batch = self._make_subhalos(100)
        halos_reference = self._make_subhalos(100)
        z_infall = np.random.uniform(self.zlens + 0.01, 4.0, 100)
        f_bound = 10 ** np.random.uniform(-2.5, -0.05, 100)
        for i in range(100):
            for h in [halos_batch[i], halos_reference[i]]:
                h._z_infall = float(z_infall[i])
                h._mbound_galacticus_definition = float(f_bound[i] * h.mass)
        precompute_tnfw_subhalos(halos_batch, self.truncation_class)
        for hb, hr in zip(halos_batch, halos_reference):
            (c_b, rt_b) = hb.profile_args
            (c_r, rt_r) = hr.profile_args
            npt.assert_almost_equal(c_b / c_r, 1, 6)
            npt.assert_almost_equal(rt_b / rt_r, 1, 6)
            npt.assert_almost_equal(hb._rescale_norm / hr._rescale_norm, 1, 6)

    def test_precompute_tnfw_bound_masses(self):

        halos_batch = self._make_subhalos(50)
        halos_reference = self._make_subhalos(50)
        z_infall = np.random.uniform(self.zlens + 0.01, 4.0, 50)
        f_bound = 10 ** np.random.uniform(-2.5, -0.05, 50)
        for i in range(50):
            for h in [halos_batch[i], halos_reference[i]]:
                h._z_infall = float(z_infall[i])
                h._mbound_galacticus_definition = float(f_bound[i] * h.mass)
        precompute_tnfw_subhalos(halos_batch, self.truncation_class)
        precompute_tnfw_bound_masses(halos_batch)
        for hb, hr in zip(halos_batch, halos_reference):
            npt.assert_almost_equal(hb.bound_mass / hr.bound_mass, 1, 6)

    def test_precompute_sidm_evolving_profiles(self):

        # the mass-conservation integral in the batch routine must use the same
        # radial resolution as TNFWCHaloEvolving.profile_args for exact agreement
        src = inspect.getsource(TNFWCHaloEvolving.profile_args.fget)
        n_r = int(re.search(r"logspace\(.*?,\s*(\d{2,5})\s*\)", src, re.S).group(1))
        np.random.seed(1)
        n = 50
        halos_batch, halos_reference = [], []
        for _ in range(n):
            cc = ConcentrationConstant(None, float(np.random.uniform(5, 20)))
            kwargs_profile = {'sidm_timescale': float(10 ** np.random.uniform(0, 1.5)),
                              'lambda_t': 1.0,
                              'mass_conservation': float(10 ** np.random.uniform(7, 10)),
                              'rt_kpc': float(np.random.uniform(5, 50))}
            args = (10 ** 8, 0.5, 1.0, None, self.zlens, False, self.lens_cosmo,
                    dict(kwargs_profile), None, cc)
            halos_batch.append(TNFWCHaloEvolving(*args, np.random.rand()))
            halos_reference.append(TNFWCHaloEvolving(*args, np.random.rand()))
        precompute_sidm_evolving_profiles(halos_batch, n_r=n_r)
        for hb, hr in zip(halos_batch, halos_reference):
            npt.assert_almost_equal(np.array(hb.profile_args) / np.array(hr.profile_args), 1, 6)
            npt.assert_almost_equal(hb.halo_effective_age / hr.halo_effective_age, 1, 6)

    def test_full_pipeline_statistical(self):

        # unpinned pipeline: random stages are drawn as arrays, so agreement
        # with the per-halo path is statistical rather than draw-for-draw
        concentration_class = ConcentrationDiemerJoyce(self.lens_cosmo.cosmo.astropy, scatter=True)
        self.concentration_class = concentration_class
        halos_batch = self._make_subhalos(500, seed=7)
        halos_reference = self._make_subhalos(500, seed=8)
        precompute_tnfw_subhalos(halos_batch, self.truncation_class)
        rt_batch = np.array([h.profile_args[1] for h in halos_batch])
        rt_reference = np.array([h.profile_args[1] for h in halos_reference])
        log_norm_batch = np.log10([h._rescale_norm for h in halos_batch])
        log_norm_reference = np.log10([h._rescale_norm for h in halos_reference])
        npt.assert_array_less(abs(np.log10(np.median(rt_batch) / np.median(rt_reference))), 0.3)
        npt.assert_array_less(abs(np.median(log_norm_batch) - np.median(log_norm_reference)), 0.3)

    def test_precompute_realization(self):

        # a realization mixing TNFW subhalos with globular clusters; unsupported
        # profile types must pass through the precompute untouched and still work
        subhalos = self._make_subhalos(50)
        globular_clusters = self._make_globular_clusters(20)
        kwargs_halo_model = {'truncation_model_subhalos': self.truncation_class,
                             'truncation_model_field_halos': self.truncation_class,
                             'concentration_model': self.concentration_class,
                             'kwargs_density_profile': {}}
        realization = Realization.from_halos(subhalos + globular_clusters, self.lens_cosmo,
                                             kwargs_halo_model, msheet_correction=False,
                                             rendering_classes=None)
        precompute_realization(realization)
        for halo in realization.halos:
            if halo.mdef == 'TNFW':
                npt.assert_equal(hasattr(halo, '_profile_args'), True)
                npt.assert_equal(hasattr(halo, '_nfw_params'), True)
            else:
                # globular clusters skipped by the batch routines
                npt.assert_equal(hasattr(halo, '_profile_args'), False)
        lens_model_list, redshift_array, kwargs_lens, _ = realization.lensing_quantities(
            add_mass_sheet_correction=False)
        npt.assert_equal(len(lens_model_list), 70)

    def test_precompute_nfw_params(self):

        nfw_halos_batch = self._make_subhalos(50, mdef='NFW') + self._make_fieldhalos(50, mdef='NFW')
        nfw_halos_reference = self._make_subhalos(50, mdef='NFW') + self._make_fieldhalos(50, mdef='NFW')
        # pin the (randomly sampled) infall redshifts so z_eval agrees between the
        # two halo populations
        z_infall = np.random.uniform(self.zlens + 0.01, 4.0, 50)
        for i in range(50):
            nfw_halos_batch[i]._z_infall = float(z_infall[i])
            nfw_halos_reference[i]._z_infall = float(z_infall[i])
        precompute_nfw_params(nfw_halos_batch)
        for hb, hr in zip(nfw_halos_batch, nfw_halos_reference):
            npt.assert_almost_equal(hb.nfw_params[0] / hr.nfw_params[0], 1, 6)
            npt.assert_almost_equal(hb.nfw_params[1] / hr.nfw_params[1], 1, 6)
            npt.assert_almost_equal(hb.nfw_params[2] / hr.nfw_params[2], 1, 6)


    def test_precompute_tnfw_field_halos(self):

        # exercise all three branches: Multiple_RS and TruncationRN (fully
        # vectorized) and the fallback that calls truncation_radius_halo per halo
        truncation_models = [Multiple_RS(self.lens_cosmo, tau=5.0),
                             TruncationRN(self.lens_cosmo, LOS_truncation_factor=50),
                             ConstantTruncationArcsec(self.lens_cosmo, 2.5)]
        for truncation_class in truncation_models:
            halos_batch, halos_reference = [], []
            np.random.seed(3)
            for _ in range(50):
                m = 10 ** np.random.uniform(6, 10.7)
                x, y = np.random.uniform(-2, 2, 2)
                z = float(np.random.choice([0.3, 0.5, 0.9]))
                args = (m, x, y, None, z, False, self.lens_cosmo, {},
                        truncation_class, self.concentration_class)
                halos_batch.append(TNFWFieldHalo(*args, np.random.rand()))
                halos_reference.append(TNFWFieldHalo(*args, np.random.rand()))
            precompute_tnfw_field_halos(halos_batch, truncation_class)
            for hb, hr in zip(halos_batch, halos_reference):
                npt.assert_equal(hasattr(hb, '_profile_args'), True)
                (c_b, rt_b) = hb.profile_args
                (c_r, rt_r) = hr.profile_args
                npt.assert_almost_equal(c_b / c_r, 1, 6)
                npt.assert_almost_equal(rt_b / rt_r, 1, 6)
                npt.assert_almost_equal(hb.nfw_params[0] / hr.nfw_params[0], 1, 8)

    def test_batch_nfw_concentration(self):

        np.random.seed(5)
        m = 10 ** np.random.uniform(6, 10.7, 100)
        z = 0.8
        # array-safe CDM model (DIEMERJOYCE19), median relation
        c_batch = batch_nfw_concentration(self.concentration_class, m, z)
        c_reference = np.array([self.concentration_class.nfw_concentration(float(mi), z) for mi in m])
        npt.assert_almost_equal(c_batch / c_reference, 1, 6)

        # WDM turnover model wrapping the CDM class: batch evaluates the CDM
        # class and applies the vectorized suppression
        wdm = ConcentrationWDMPolynomial(self.lens_cosmo.cosmo.astropy, ConcentrationDiemerJoyce,
                                         log_mc=7.5, kwargs_cdm={'scatter': False})
        c_batch = batch_nfw_concentration(wdm, m, z)
        c_reference = np.array([wdm.nfw_concentration(float(mi), z) for mi in m])
        npt.assert_almost_equal(c_batch / c_reference, 1, 6)

        # constant model
        c_const = ConcentrationConstant(None, 9.0)
        c_batch = batch_nfw_concentration(c_const, m, z)
        npt.assert_almost_equal(c_batch, 9.0, 6)
        npt.assert_equal(len(c_batch), len(m))

        # unknown custom model: falls back to the per-halo path
        class CustomConcentration(object):
            name = 'CUSTOM'
            def nfw_concentration(self, m_, z_):
                return 5.0 + np.log10(m_)
        custom = CustomConcentration()
        c_batch = batch_nfw_concentration(custom, m, z)
        c_reference = np.array([custom.nfw_concentration(float(mi), z) for mi in m])
        npt.assert_almost_equal(c_batch / c_reference, 1, 6)

        # scatter: one vectorized lognormal draw with the same distribution as
        # the per-halo draws; check the median tracks the median relation and
        # the scatter amplitude matches scatter_dex
        cmodel_scatter = ConcentrationDiemerJoyce(self.lens_cosmo.cosmo.astropy,
                                                  scatter=True, scatter_dex=0.2)
        m_fixed = np.full(20000, 10 ** 8)
        c_scattered = batch_nfw_concentration(cmodel_scatter, m_fixed, z)
        c_median = cmodel_scatter.nfw_concentration(10 ** 8.0, z, force_no_scatter=True)
        npt.assert_array_less(abs(np.median(c_scattered) / c_median - 1), 0.02)
        npt.assert_array_less(abs(np.std(np.log(c_scattered)) / 0.2 - 1), 0.05)


if __name__ == '__main__':
    pytest.main()
