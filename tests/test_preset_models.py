from pyHalo.PresetModels.cdm import CDM
from pyHalo.PresetModels.wdm import WDM, WDM_mixed
from pyHalo.PresetModels.sidm import SIDM_core_collapse
from pyHalo.PresetModels.uldm import ULDM
from pyHalo.preset_models import preset_model_from_name
from pyHalo.PresetModels.external import CDMFromEmulator, DMFromGalacticus
from pyHalo.Halos.galacticus_util.galacticus_util import GalacticusUtil
from pyHalo.Halos.HaloModels.TNFWFromParams import TNFWFromParams
import pytest
import numpy as np
import numpy.testing as npt


class TestPresetModels(object):

    def test_CDM(self):

        cdm = CDM(0.5, 1.5)
        _ = cdm.lensing_quantities()
        _ = preset_model_from_name('CDM')

        cdm = CDM(0.5, 1.5,
                  truncation_model_subhalos='TRUNCATION_GALACTICUS')

    def test_WDM(self):

        wdm = WDM(0.5, 1.5, 8.0)
        _ = wdm.lensing_quantities()
        _ = preset_model_from_name('WDM')

        _ = WDM(0.5, 1.5, 8.0,
                  truncation_model_subhalos='TRUNCATION_GALACTICUS')

    def test_ULDM(self):

        flucs_shape = 'ring'
        flucs_args = {'angle': 0.0, 'rmin': 0.9, 'rmax': 1.1}
        uldm = ULDM(0.5, 1.5, -21, flucs_shape=flucs_shape, flucs_args=flucs_args)
        _ = uldm.lensing_quantities()
        _ = preset_model_from_name('ULDM')

    def test_SIDM_core_collapse(self):
        mass_ranges_subhalos = [[6, 8], [8, 10]]
        mass_ranges_field_halos = [[6, 8], [8, 10]]
        probabilities_subhalos = [1, 1]
        probabilities_field_halos = [1, 1]
        sidm_cc = SIDM_core_collapse(0.5, 1.5, mass_ranges_subhalos, mass_ranges_field_halos,
        probabilities_subhalos, probabilities_field_halos)
        _ = sidm_cc.lensing_quantities()
        _ = preset_model_from_name('SIDM_core_collapse')

    def test_WDM_mixed(self):
        wdm_mixed = WDM_mixed(0.5, 1.5, 8.0, 0.5)
        _ = wdm_mixed.lensing_quantities()
        _ = preset_model_from_name('WDM_mixed')

    def test_WDM_general(self):
        func = preset_model_from_name('WDMGeneral')
        wdm = func(0.5, 1.5, 7.7, 2.0)
        _ = wdm.lensing_quantities()

        wdm = func(0.5, 1.5, 7.7, 2.0, truncation_model_subhalos='TRUNCATION_GALACTICUS')

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

        test_data = {
            util.PARAM_X:                       np.asarray((1,1,0,1,0 ,1,0 ,0,0,1)), 
            util.PARAM_Y:                       np.asarray((1,0,1,0,0 ,1,0 ,0,3,0)),
            util.PARAM_Z:                       np.asarray((0,1,1,0,0 ,0,0 ,0,0,0)),
            util.PARAM_TNFW_RHO_S:              np.asarray((1,1,1,1,1 ,1,1 ,1,1,1)),
            util.PARAM_TNFW_RADIUS_TRUNCATION:  np.asarray((1,1,1,1,1 ,1,1 ,1,1,1)),
            util.PARAM_RADIUS_VIRIAL:           np.asarray((2,2,2,2,2 ,2,2 ,2,2,2)),
            util.PARAM_RADIUS_SCALE:            np.asarray((1,1,1,1,2 ,1,2 ,2,2,2)),
            util.PARAM_MASS_BOUND:              np.asarray((1,2,3,4,2 ,1,2 ,2,1,1)),
            util.PARAM_MASS_BASIC:              np.asarray((5,6,7,1,2 ,1,2 ,2,1,1)),
            util.PARAM_ISOLATED:                np.asarray((0,0,0,0,1 ,0,1 ,1,0,0)),
            util.PARAM_TREE_ORDER:              np.asarray((0,0,0,0,0 ,1,1 ,2,2,2)),
            util.PARAM_TREE_INDEX:              np.asarray((1,1,1,1,1 ,2,2 ,3,3,3)),
            util.PARAM_NODE_ID:                 np.asarray((0,1,2,3,4 ,5,6 ,7,8,9))
        }

        kwargs_realization_base = dict(
            z_lens=                             0.5,
            z_source=                           2,
            galacticus_hdf5=                    test_data,
            tree_index=                         1,
            kwargs_cdm=                         dict(cone_opening_angle_arcsec=1E10),
            mass_range=                         (0,10),
            mass_range_is_bound=                False, 
            proj_plane_normal=                  np.asarray((1,0,0)),
            include_field_halos=                False,
            nodedata_filter=                    None,
            galacticus_utilities=               util,
            galacticus_params_additional=       None, 
            proj_rotation_angles=               None,
            tabulate_radius_truncation=         None
        )

        realization_test_projection = DMFromGalacticus(**kwargs_realization_base)

        assert len(realization_test_projection.halos) == 4

        MPC_TO_KPC = 1E3
        MPC_TO_AS = MPC_TO_KPC / realization_test_projection.lens_cosmo.cosmo.kpc_proper_per_asec(kwargs_realization_base["z_lens"]) 
    
        npt.assert_almost_equal(np.linalg.norm(np.asarray((realization_test_projection.halos[0].x,realization_test_projection.halos[0].y))), MPC_TO_AS * 1)
        npt.assert_almost_equal(np.linalg.norm(np.asarray((realization_test_projection.halos[1].x,realization_test_projection.halos[1].y))), MPC_TO_AS * 1)
        npt.assert_almost_equal(np.linalg.norm(np.asarray((realization_test_projection.halos[2].x,realization_test_projection.halos[2].y))), MPC_TO_AS * np.sqrt(2))
        npt.assert_almost_equal(np.linalg.norm(np.asarray((realization_test_projection.halos[3].x,realization_test_projection.halos[3].y))), 0)

        realization_test_filter_mass_basic = DMFromGalacticus(**(kwargs_realization_base | dict(mass_range=(4.99,6.01),mass_range_is_bound=False)))
        npt.assert_equal(len(realization_test_filter_mass_basic.halos),2)

        realization_test_filter_mass_bound = DMFromGalacticus(**(kwargs_realization_base | dict(mass_range=(0.99,2.01),mass_range_is_bound=True)))
        npt.assert_equal(len(realization_test_filter_mass_bound.halos),2)

        kwargs_realization_test_volume_exclusion = dict(
            kwargs_cdm=                         dict(cone_opening_angle_arcsec= 0.99 * MPC_TO_AS * 2), 
        )

        realization_test_volume_exclusion = DMFromGalacticus(**(kwargs_realization_base | kwargs_realization_test_volume_exclusion))

        npt.assert_equal(len(realization_test_volume_exclusion.halos),1)

        realization_test_tree2 = DMFromGalacticus(**(kwargs_realization_base | dict(tree_index=2)))

        npt.assert_equal(len(realization_test_tree2.halos),1)

        realization_test_exclude_beyond_virial = DMFromGalacticus(**(kwargs_realization_base | dict(tree_index=3)))

        npt.assert_equal(len(realization_test_exclude_beyond_virial.halos),1)

        realization_test_params_physical = DMFromGalacticus(**(kwargs_realization_base | dict(nodedata_filter = lambda nd,u: np.ones(nd[u.PARAM_MASS_BASIC].shape[0],dtype=bool))))

        for n,sh in enumerate(realization_test_params_physical.halos):    
            npt.assert_almost_equal(sh.params_physical[TNFWFromParams.KEY_RHO_S],test_data[util.PARAM_TNFW_RHO_S][n] * 4 * 1 / MPC_TO_KPC**3)
            npt.assert_almost_equal(sh.params_physical[TNFWFromParams.KEY_RS],test_data[util.PARAM_RADIUS_SCALE][n] * MPC_TO_KPC)
            npt.assert_almost_equal(sh.params_physical[TNFWFromParams.KEY_RV],test_data[util.PARAM_RADIUS_VIRIAL][n] * MPC_TO_KPC)
            npt.assert_almost_equal(sh.params_physical[TNFWFromParams.KEY_RT],test_data[util.PARAM_TNFW_RADIUS_TRUNCATION][n] * MPC_TO_KPC)



if __name__ == '__main__':
     pytest.main()
