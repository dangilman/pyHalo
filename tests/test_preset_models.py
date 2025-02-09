from pyHalo.PresetModels.cdm import CDM, CDMCorrelatedStructure
from pyHalo.PresetModels.wdm import WDM, WDM_mixed
from pyHalo.PresetModels.sidm import SIDM_core_collapse, SIDM_parametric
from pyHalo.PresetModels.uldm import ULDM
from pyHalo.preset_models import preset_model_from_name
from pyHalo.PresetModels.external import CDMFromEmulator, DMFromGalacticus
from pyHalo.Halos.galacticus_util.galacticus_util import GalacticusUtil
from pyHalo.Halos.HaloModels.TNFWFromParams import TNFWFromParams
import pytest
import numpy as np
import numpy.testing as npt
from copy import copy
from pyHalo.Cosmology.cosmology import Cosmology


class TestPresetModels(object):

    def _test_default_infall_model(self, realization, default):

        lens_cosmo = realization.lens_cosmo
        infall_redshift_model = lens_cosmo._z_infall_model
        npt.assert_string_equal(default, infall_redshift_model.name)

    def test_CDM(self):

        cdm = CDM(0.5, 1.5)
        _ = cdm.lensing_quantities()
        _ = preset_model_from_name('CDM')

        cdm2 = CDM(0.6, 1.5, log_m_host=13.3,
                  truncation_model_subhalos='TRUNCATION_GALACTICUS')
        cdm3 = CDM(0.6, 1.5, log_m_host=13.3,
                   truncation_model_subhalos='TRUNCATION_GALACTICUS',
                   host_scaling_factor=4.0, redshift_scaling_factor=2.0,
                   infall_redshift_model='DIRECT_INFALL', kwargs_infall_model={})
        halos2 = len(cdm2.halos)
        halos3 = len(cdm3.halos)
        npt.assert_equal(halos3 > halos2, True)

        self._test_default_infall_model(cdm, 'hybrid')
        self._test_default_infall_model(cdm3, 'direct')

        kwargs_globular_clusters = {'log10_mgc_mean': 5.0,
                                    'log10_mgc_sigma': 0.5,
                                    'rendering_radius_arcsec': 0.1,
                                    }
        cdm = CDM(0.5, 1.5,
                  add_globular_clusters=True,
                  kwargs_globular_clusters=kwargs_globular_clusters)
        _ = cdm.lensing_quantities()

    def test_CDM_correlated_structure_only(self):

        cdm = CDMCorrelatedStructure(0.5, 1.5)
        _ = cdm.lensing_quantities()

    def test_WDM(self):

        wdm = WDM(0.5, 1.5, 8.0)
        _ = wdm.lensing_quantities()
        _ = preset_model_from_name('WDM')
        self._test_default_infall_model(wdm, 'hybrid')
        kwargs_globular_clusters = {'log10_mgc_mean': 5.0,
                                    'log10_mgc_sigma': 0.5,
                                    'rendering_radius_arcsec': 0.1,
                                    }
        wdm = WDM(0.5, 1.5,
                  7.7,
                  add_globular_clusters=True,
                  kwargs_globular_clusters=kwargs_globular_clusters)
        _ = wdm.lensing_quantities()

    def test_SIDM(self):

        mass_ranges = [[6, 8], [8, 10]]
        collapse_times = [10.5, 1.0]
        sidm = SIDM_parametric(0.5, 1.5, mass_ranges, collapse_times)
        _ = sidm.lensing_quantities()
        _ = preset_model_from_name('SIDM_parametric')
        self._test_default_infall_model(sidm, 'hybrid')

        model = preset_model_from_name('SIDM_parametric_fixedbins')
        realization = model(0.5, 1.5, 5.0, 1.0)
        _ = realization.lensing_quantities()

    def test_ULDM(self):

        flucs_shape = 'ring'
        flucs_args = {'angle': 0.0, 'rmin': 0.9, 'rmax': 1.1}
        uldm = ULDM(0.5, 1.5, -21, flucs_shape=flucs_shape, flucs_args=flucs_args)
        _ = uldm.lensing_quantities()
        _ = preset_model_from_name('ULDM')
        self._test_default_infall_model(uldm, 'hybrid')

    def test_SIDM_core_collapse(self):
        mass_ranges_subhalos = [[6, 8], [8, 10]]
        mass_ranges_field_halos = [[6, 8], [8, 10]]
        probabilities_subhalos = [1, 1]
        probabilities_field_halos = [1, 1]
        sidm_cc = SIDM_core_collapse(0.5, 1.5, mass_ranges_subhalos, mass_ranges_field_halos,
        probabilities_subhalos, probabilities_field_halos)
        _ = sidm_cc.lensing_quantities()
        _ = preset_model_from_name('SIDM_core_collapse')
        self._test_default_infall_model(sidm_cc, 'hybrid')

    def test_WDM_mixed(self):
        wdm_mixed = WDM_mixed(0.5, 1.5, 8.0, 0.5)
        _ = wdm_mixed.lensing_quantities()
        _ = preset_model_from_name('WDM_mixed')
        self._test_default_infall_model(wdm_mixed, 'hybrid')

    def test_CDM_blackholes(self):

        model = preset_model_from_name('CDM_plus_BH')
        cdm_bh = model(0.5,
                             1.5,
                             -0.2,
                             -0.3,
                             sigma_sub=0.01)
        _ = cdm_bh.lensing_quantities()
        _ = preset_model_from_name('CDM_plus_BH')

    def test_WDM_general(self):
        func = preset_model_from_name('WDMGeneral')
        wdm = func(0.5, 1.5, 7.7, -2.0)
        _ = wdm.lensing_quantities()
        wdm = func(0.5, 1.5, 7.7, -2.0, truncation_model_subhalos='TRUNCATION_GALACTICUS')
        self._test_default_infall_model(wdm, 'hybrid')
        kwargs_globular_clusters = {'log10_mgc_mean': 5.0,
                                    'log10_mgc_sigma': 0.5,
                                    'rendering_radius_arcsec': 0.1,
                                    }
        wdm = func(0.5, 1.5, 7.7, -2.5,
                  add_globular_clusters=True,
                  kwargs_globular_clusters=kwargs_globular_clusters)
        _ = wdm.lensing_quantities()

    def test_CDM_emulator(self):

        def emulator_input_callable(*args, **kwargs):
            subhalo_infall_masses = np.array([10**7,10**8])
            subhalo_x_kpc = np.array([1.0, 1.0])
            subhalo_y_kpc = np.array([1.0, 1.0])
            subhalo_final_bound_masses = subhalo_infall_masses / 2
            subhalo_infall_concentrations = np.array([16.0, 20.0])
            return subhalo_infall_masses, subhalo_x_kpc, subhalo_y_kpc, subhalo_final_bound_masses, subhalo_infall_concentrations

        concentrations = np.array([16.0, 20.0])
        mass_array = np.array([10 ** 7, 10 ** 8])
        kwargs_cdm = {'LOS_normalization': 0.0}
        cdm_subhalo_emulator = CDMFromEmulator(0.5, 1.5, emulator_input_callable, kwargs_cdm)
        _ = cdm_subhalo_emulator.lensing_quantities()
        for i, halo in enumerate(cdm_subhalo_emulator.halos):
            npt.assert_equal(halo.mass, mass_array[i])
            npt.assert_almost_equal(halo.x, 0.1584666, 4)
            npt.assert_almost_equal(halo.y, 0.1584666, 4)
            npt.assert_equal(halo.c, concentrations[i])

        emulator_input_array = np.empty((2, 5))
        emulator_input_array[:, 0] = mass_array
        emulator_input_array[:, 1] = np.array([1.0, 1.0])
        emulator_input_array[:, 2] = np.array([1.0, 1.0])
        emulator_input_array[:, 3] = mass_array / 2
        emulator_input_array[:, 4] = concentrations
        cdm_subhalo_emulator = CDMFromEmulator(0.5, 1.5, emulator_input_array, kwargs_cdm)
        _ = cdm_subhalo_emulator.lensing_quantities()
        for i, halo in enumerate(cdm_subhalo_emulator.halos):
            npt.assert_equal(halo.mass, mass_array[i])
            npt.assert_almost_equal(halo.x, 0.1584666, 4)
            npt.assert_almost_equal(halo.y, 0.1584666, 4)
            npt.assert_equal(halo.c, concentrations[i])

    def test_galacticus(self):
        util = GalacticusUtil()

        cosmo = Cosmology()

        # Simulates data loaded from a galacticus hdf5 file.
        # mock_data is in NOT intended to be physical
        # It is however designed to test aspects of DMfromGalacticus
        mock_data = {
            util.PARAM_X:                       np.asarray((1,1,0,1,0  ,  1,0  ,   0,1,0  )),
            util.PARAM_Y:                       np.asarray((1,0,1,0,0  ,  1,0  ,   3,0,0  )),
            util.PARAM_Z:                       np.asarray((0,1,1,0,0  ,  0,0  ,   0,0,0  )),
            util.PARAM_TNFW_RHO_S:              np.asarray((1,1,1,1,1  ,  1,1  ,   1,1,1  )),
            util.PARAM_TNFW_RADIUS_TRUNCATION:  np.asarray((1,1,1,1,1  ,  1,1  ,   1,1,1  )),
            util.PARAM_RADIUS_VIRIAL:           np.asarray((2,2,2,2,2  ,  2,2  ,   2,2,2  )),
            util.PARAM_RADIUS_SCALE:            np.asarray((1,1,1,1,2  ,  1,2  ,   2,2,2  )),
            util.PARAM_MASS_BOUND:              np.asarray((1,2,3,4,2  ,  1,2  ,   1,1,2  )),
            util.PARAM_MASS_INFALL:             np.asarray((5,6,7,1,2  ,  1,2  ,   1,1,2  )),
            util.PARAM_ISOLATED:                np.asarray((0,0,0,0,1  ,  0,1  ,   0,0,1  )),
            util.PARAM_TREE_ORDER:              np.asarray((0,0,0,0,0  ,  1,1  ,   2,2,2  )),
            util.PARAM_TREE_INDEX:              np.asarray((1,1,1,1,1  ,  2,2  ,   3,3,3  )),
            util.PARAM_NODE_ID:                 np.asarray((0,1,2,3,4  ,  5,6  ,   8,9,7  )),
            util.PARAM_Z_LAST_ISOLATED:         np.asarray((1,2,3,2,0.5,  1,0.5,   3,1,0.5))
        }

        mock_data[util.PARAM_CONCENTRATION] = mock_data[util.PARAM_RADIUS_VIRIAL] / mock_data[util.PARAM_RADIUS_SCALE]

        kwargs_base = dict(
            galacticus_hdf5=                        mock_data,
            z_source=                               2,
            cone_opening_angle_arcsec=              1E10,
            tree_index=                             1,
            log_mlow_galacticus=                    -10,
            log_mhigh_galacticus=                   10,
            mass_range_is_bound=                    False,
            proj_angle_theta=                       np.pi/2,
            proj_angle_phi=                         0,
            nodedata_filter=                        None,
            galacticus_utilities=                   util,
            galacticus_params_additional=           None,
            galacticus_tabulate_tnfw_params=        None,
            preset_model_los=                       "CDM",
            LOS_normalization=                      0.0
        )

        MPC_TO_KPC = 1E3
        MPC_TO_AS = MPC_TO_KPC / cosmo.kpc_proper_per_asec(0.5)

        kwargs_test_projection = copy(kwargs_base)
        realization_test_projection = DMFromGalacticus(**kwargs_test_projection)
        assert len(realization_test_projection.halos) == 4
        npt.assert_almost_equal(np.linalg.norm(np.asarray((realization_test_projection.halos[0].x,realization_test_projection.halos[0].y))), MPC_TO_AS * 1)
        npt.assert_almost_equal(np.linalg.norm(np.asarray((realization_test_projection.halos[1].x,realization_test_projection.halos[1].y))), MPC_TO_AS * 1)
        npt.assert_almost_equal(np.linalg.norm(np.asarray((realization_test_projection.halos[2].x,realization_test_projection.halos[2].y))), MPC_TO_AS * np.sqrt(2))
        npt.assert_almost_equal(np.linalg.norm(np.asarray((realization_test_projection.halos[3].x,realization_test_projection.halos[3].y))), 0)

        kwargs_test_filter_mass_basic = copy(kwargs_base)
        kwargs_test_filter_mass_basic.update(dict(log_mlow_galacticus=np.log10(4.99),log_mhigh_galacticus=np.log10(6.01)
                                                  ,mass_range_is_bound=False))
        realization_test_filter_mass_basic = DMFromGalacticus(**kwargs_test_filter_mass_basic)
        npt.assert_equal(len(realization_test_filter_mass_basic.halos),2)

        kwargs_test_filter_mass_bound = copy(kwargs_base)
        kwargs_test_filter_mass_bound.update(dict(log_mlow_galacticus=np.log10(0.99),log_mhigh_galacticus=np.log10(2.01)
                                                ,mass_range_is_bound=True))
        realization_test_filter_mass_bound = DMFromGalacticus(**kwargs_test_filter_mass_bound)
        npt.assert_equal(len(realization_test_filter_mass_bound.halos),2)

        kwargs_test_volume_exclusion = copy(kwargs_base)
        kwargs_test_volume_exclusion["cone_opening_angle_arcsec"] = 0.99 * MPC_TO_AS * 2
        realization_test_volume_exclusion = DMFromGalacticus(**kwargs_test_volume_exclusion)
        npt.assert_equal(len(realization_test_volume_exclusion.halos),1)

        kwargs_test_tree2 = copy(kwargs_base)
        kwargs_test_tree2["tree_index"] = 2
        realization_test_tree2 = DMFromGalacticus(**kwargs_test_tree2)
        npt.assert_equal(len(realization_test_tree2.halos),1)

        kwargs_test_exclude_beyond_virial = copy(kwargs_base)
        kwargs_test_exclude_beyond_virial["tree_index"] = 3
        realization_test_exclude_beyond_virial = DMFromGalacticus(**kwargs_test_exclude_beyond_virial)
        npt.assert_equal(len(realization_test_exclude_beyond_virial.halos),1)

        kwargs_test_params = copy(kwargs_base)
        kwargs_test_params["nodedata_filter"] = lambda nd,u: np.ones(nd[u.PARAM_MASS_INFALL].shape[0],dtype=bool)
        realization_test_params = DMFromGalacticus(**kwargs_test_params)
        for n,sh in enumerate(realization_test_params.halos):
            npt.assert_almost_equal(sh.params_physical[TNFWFromParams.KEY_RHO_S],mock_data[util.PARAM_TNFW_RHO_S][n] * 4 * 1 / MPC_TO_KPC**3)
            npt.assert_almost_equal(sh.params_physical[TNFWFromParams.KEY_RS],mock_data[util.PARAM_RADIUS_SCALE][n] * MPC_TO_KPC)
            npt.assert_almost_equal(sh.params_physical[TNFWFromParams.KEY_RV],mock_data[util.PARAM_RADIUS_VIRIAL][n] * MPC_TO_KPC)
            npt.assert_almost_equal(sh.params_physical[TNFWFromParams.KEY_RT],mock_data[util.PARAM_TNFW_RADIUS_TRUNCATION][n] * MPC_TO_KPC)
            npt.assert_almost_equal(sh.z,0.5)
            npt.assert_almost_equal(sh.z_infall,mock_data[util.PARAM_Z_LAST_ISOLATED][n])

if __name__ == '__main__':
    pytest.main()
