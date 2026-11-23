import pytest
import numpy as np
import numpy.testing as npt
from pyHalo.preset_models import preset_model_from_name
from pyHalo.PresetModels.cdm import CDM
from pyHalo.PresetModels.nodes import HMFConcentrationNodes
from pyHalo.Halos.concentration_node_ratio import ConcentrationNodeRatio


class TestHMFConcentrationNodes(object):

    def setup_method(self):

        # without Poisson draws the number of halos is set by the mass functions
        self.kwargs = {'z_lens': 0.5, 'z_source': 2.0, 'cone_opening_angle_arcsec': 4.0, 'draw_poisson': False}

    @staticmethod
    def _counts(realization):
        subhalo = np.array([halo.is_subhalo for halo in realization.halos])
        return subhalo.sum(), (~subhalo).sum()

    def test_registered(self):

        npt.assert_equal(preset_model_from_name('HMFConcentrationNodes') is HMFConcentrationNodes, True)

    def test_cdm(self):

        # with every node at zero the mass functions are those of the CDM preset
        npt.assert_equal(self._counts(HMFConcentrationNodes(**self.kwargs)), self._counts(CDM(**self.kwargs)))

    def test_mass_function_nodes(self):

        n_sub_cdm, n_field_cdm = self._counts(CDM(**self.kwargs))
        n_sub, n_field = self._counts(HMFConcentrationNodes(log10_n_ratio_6=0.4, log10_n_ratio_8=0.2, **self.kwargs))
        npt.assert_array_less([n_sub_cdm, n_field_cdm], [n_sub, n_field])
        n_sub, n_field = self._counts(HMFConcentrationNodes(log10_n_ratio_6=-1.0, log10_n_ratio_8=-0.5,
                                                            **self.kwargs))
        npt.assert_array_less([n_sub, n_field], [n_sub_cdm, n_field_cdm])

    def test_factor_of_two(self):

        # with every node at log10(2), the model is CDM with twice as many halos at every mass (subhalos and field
        # halos) and concentrations twice the CDM concentrations; check this on halos sampled from both models
        log10_2 = np.log10(2.0)
        kwargs = dict(self.kwargs, cone_opening_angle_arcsec=6.0)
        np.random.seed(11)
        realization_cdm = CDM(**kwargs)
        np.random.seed(12)
        realization = HMFConcentrationNodes(log10_n_ratio_6=log10_2, log10_n_ratio_8=log10_2, log10_n_ratio_10=log10_2,
                                            log10_c_ratio_6=log10_2, log10_c_ratio_8=log10_2, log10_c_ratio_10=log10_2,
                                            **kwargs)

        # 1) the number of halos: without Poisson draws, exact up to rounding (once for subhalos, on each lens plane
        # for field halos)
        n_sub_cdm, n_field_cdm = self._counts(realization_cdm)
        n_sub, n_field = self._counts(realization)
        npt.assert_allclose(n_sub / n_sub_cdm, 2.0, rtol=2e-3)
        npt.assert_allclose(n_field / n_field_cdm, 2.0, rtol=2e-2)

        # 2) the sampled mass functions: twice as many halos in every bin of halo mass, within the sampling noise of the
        # masses (4 sigma)
        bins = np.linspace(6.0, 10.0, 9)
        for is_subhalo in [True, False]:
            log10m_cdm = np.log10([h.mass for h in realization_cdm.halos if h.is_subhalo == is_subhalo])
            log10m = np.log10([h.mass for h in realization.halos if h.is_subhalo == is_subhalo])
            counts_cdm = np.histogram(log10m_cdm, bins)[0]
            counts = np.histogram(log10m, bins)[0]
            sigma = np.sqrt(counts + 4 * counts_cdm)
            npt.assert_array_less(np.abs(counts - 2 * counts_cdm), 4 * sigma + 1)

        # 3) the sampled concentrations: log10(c / c_CDM,median) at each halo's mass and redshift is shifted by log10(2),
        # with the same scatter as in CDM
        def log10_c_residual(realization, cdm_class_of):
            residual = []
            for halo in realization.halos[0:3000]:
                c_median = cdm_class_of(halo).nfw_concentration(halo.mass, halo.z_eval, force_no_scatter=True)
                residual.append(np.log10(halo.c / c_median))
            return np.array(residual)
        residual_cdm = log10_c_residual(realization_cdm, lambda halo: halo._concentration_class)
        residual = log10_c_residual(realization, lambda halo: halo._concentration_class._cdm_concentration)
        npt.assert_allclose(np.mean(residual) - np.mean(residual_cdm), log10_2, atol=0.01)
        npt.assert_allclose(np.std(residual), np.std(residual_cdm), rtol=0.05)
        npt.assert_allclose(np.std(residual), 0.15, rtol=0.05)

    def test_concentration_nodes(self):

        realization = HMFConcentrationNodes(log10_c_ratio_6=0.5, log10_c_ratio_8=0.3, log10_c_ratio_10=0.1,
                                            **self.kwargs)
        for halo in realization.halos[0:5]:
            npt.assert_equal(isinstance(halo._concentration_class, ConcentrationNodeRatio), True)
            # concentrations are evaluated at z_eval (the infall redshift for subhalos)
            c_cdm = halo._concentration_class._cdm_concentration.nfw_concentration(halo.mass, halo.z_eval,
                                                                                    force_no_scatter=True)
            # within 5 sigma of the scatter around the rescaled CDM median
            ratio = np.log10(halo.c / c_cdm) - np.log10(halo._concentration_class.suppression(halo.mass, halo.z_eval))
            npt.assert_array_less(abs(ratio), 5 * 0.15)

    def test_lensing_quantities(self):

        realization = HMFConcentrationNodes(log10_n_ratio_6=0.3, log10_c_ratio_6=0.2, **self.kwargs)
        lens_model_names, redshifts, kwargs_lens, _ = realization.lensing_quantities()
        npt.assert_equal(len(lens_model_names), len(kwargs_lens))
        npt.assert_equal('CONVERGENCE' in lens_model_names, True)

    def test_reserved_keywords(self):

        npt.assert_raises(TypeError, HMFConcentrationNodes, concentration_model_subhalos='LUDLOW2016', **self.kwargs)


if __name__ == '__main__':
    pytest.main()
