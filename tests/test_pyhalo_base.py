import pytest
import numpy.testing as npt
from pyHalo.pyhalo import pyHalo
from pyHalo.defaults import cosmo_default
from pyHalo.Cosmology.geometry import Geometry
from pyHalo.Rendering.MassFunctions.mass_function_base import CDMPowerLaw
from pyHalo.Rendering.MassFunctions.density_peaks import ShethTormen
from pyHalo.Rendering.SpatialDistributions.uniform import LensConeUniform
from pyHalo.Rendering.SpatialDistributions.nfw import ProjectedNFW
from pyHalo.Halos.tidal_truncation import TruncationRN, TruncationRoche
from pyHalo.Halos.concentration import ConcentrationDiemerJoyce

class TestpyHalo(object):

    def setup_method(self):
        pass
        #self.pyhalo = pyHalo(0.5, 2.)

    def test_create_realizations(self):
        pyhalo = pyHalo(0.5, 2.)
        cone_opening_angle = 6.0
        geometry = Geometry(pyhalo.cosmology, pyhalo.zlens, pyhalo.zsource, cone_opening_angle, 'DOUBLE_CONE')
        population_model_list = ['SUBHALOS', 'LINE_OF_SIGHT', 'LINE_OF_SIGHT_NOSHEET', 'TWO_HALO']
        mass_function_class_list = [CDMPowerLaw, ShethTormen, ShethTormen, ShethTormen]
        kwargs_subhalos = {'log_mlow': 6.0, 'log_mhigh': 10.0, 'm_pivot': 10**8, 'power_law_index': -1.9,
                           'delta_power_law_index': 0.0, 'draw_poisson': False, 'log_m_host': 13.3,
                           'sigma_sub': 0.05}
        kwargs_los = {'log_mlow': 6.0, 'log_mhigh': 10.0, 'm_pivot': 1e8,
                      'LOS_normalization': 1.0, 'delta_power_law_index': 0.0}
        kwargs_two_halo = {'log_mlow': 6.0, 'log_mhigh': 10.0, 'm_pivot': 1e8,
                           'LOS_normalization': 1.0, 'delta_power_law_index': 0.0, 'log_m_host': 13.0}
        kwargs_mass_function_list = [kwargs_subhalos, kwargs_los, kwargs_los, kwargs_two_halo]
        spatial_distribution_class_list = [ProjectedNFW, LensConeUniform, LensConeUniform, LensConeUniform]
        kwargs_subhalos_spatial = {'m_host': 10 ** 13.0, 'zlens': 0.5, 'rmax2d_arcsec': 3.0, 'r_core_units_rs': 0.2,
                                   'lens_cosmo': pyhalo.lens_cosmo}
        kwargs_los_spatial = {'cone_opening_angle': 6.0, 'geometry': geometry}
        kwargs_spatial_distribution_list = [kwargs_subhalos_spatial, kwargs_los_spatial,
                                       kwargs_los_spatial, kwargs_los_spatial]
        mdef_subhalos = 'TNFW'
        mdef_field_halos = 'TNFW'
        kwargs_halo_model = {'truncation_model_subhalos': TruncationRoche(pyhalo.lens_cosmo),
                             'concentration_model_subhalos': ConcentrationDiemerJoyce(pyhalo.lens_cosmo),
                             'truncation_model_field_halos': TruncationRN(pyhalo.lens_cosmo),
                             'concentration_model_field_halos': ConcentrationDiemerJoyce(pyhalo.lens_cosmo),
                             'kwargs_density_profile': {}}
        realization_list = pyhalo.render(population_model_list,
                                         mass_function_class_list,
                                         kwargs_mass_function_list,
                                         spatial_distribution_class_list,
                                         kwargs_spatial_distribution_list,
               geometry, mdef_subhalos, mdef_field_halos, kwargs_halo_model, nrealizations=2)
        npt.assert_equal(len(realization_list), 2)

    def test_cosmology_setup_from_default(self):
        pyhalo = pyHalo(0.5, 2.)
        astropy = pyhalo.astropy_cosmo
        npt.assert_equal(astropy.h * 100, cosmo_default('H0'))
        npt.assert_equal(astropy.Om0, cosmo_default('Om0'))
        npt.assert_equal(astropy.Ob0, cosmo_default('Ob0'))
        if cosmo_default.cosmo_param_dictionary['flat']:
            npt.assert_equal(astropy.Om0 + astropy.Ode0, 1.0)
        else:
            npt.assert_equal(False, astropy.Om0 + astropy.Ode0!=1.0)

    def test_cosmology_setup_from_kwargs(self):

        kwargs_cosmo = {'H0': 70.0,
                        'Ob0': 0.05,
                        'Om0': 0.25,
                        'sigma8': 0.81,
                        'flat': True,
                        'ns': 0.965,
                        'power_law': False}

        pyhalo = pyHalo(0.5, 2.0, kwargs_cosmo)
        astropy = pyhalo.astropy_cosmo
        npt.assert_equal(astropy.h * 100, kwargs_cosmo['H0'])
        npt.assert_equal(astropy.Om0, kwargs_cosmo['Om0'])
        npt.assert_equal(astropy.Ob0, kwargs_cosmo['Ob0'])
        npt.assert_equal(astropy.Om0 + astropy.Ode0, 1.0)
        colosus_cosmo = pyhalo.cosmology.colossus
        npt.assert_equal(colosus_cosmo.flat, kwargs_cosmo['flat'])
        npt.assert_equal(colosus_cosmo.Om0, kwargs_cosmo['Om0'])
        npt.assert_equal(colosus_cosmo.Ob0, kwargs_cosmo['Ob0'])
        npt.assert_equal(colosus_cosmo.sigma8, kwargs_cosmo['sigma8'])
        npt.assert_equal(colosus_cosmo.ns, kwargs_cosmo['ns'])

    def test_cosmology_dependence(self):

        # make sure that boosting stuff like total amount of dark matter actually increases the number of halos
        mdef_subhalos = 'TNFW'
        mdef_field_halos = 'TNFW'
        population_model_list = ['LINE_OF_SIGHT']
        kwargs_los = {'log_mlow': 6.0, 'log_mhigh': 10.0, 'm_pivot': 1e8,
                      'LOS_normalization': 1.0, 'delta_power_law_index': 0.0}
        kwargs_mass_function_list = [kwargs_los]
        spatial_distribution_class_list = [LensConeUniform]
        mass_function_class_list = [ShethTormen]
        kwargs_cosmo_1 = {'H0': 70.0,
                        'Ob0': 0.05,
                        'Om0': 0.2,
                        'sigma8': 1.0,
                        'flat': True,
                        'ns': 0.965,
                        'power_law': False}
        pyhalo_1 = pyHalo(0.5, 2.0, kwargs_cosmo_1)
        cone_opening_angle = 6.0
        geometry_1 = Geometry(pyhalo_1.cosmology, pyhalo_1.zlens, pyhalo_1.zsource, cone_opening_angle,
                              'DOUBLE_CONE')

        kwargs_spatial_distribution_list_1 = [{'cone_opening_angle': 6.0, 'geometry': geometry_1}]
        kwargs_halo_model_1 = {'truncation_model_subhalos': TruncationRoche(pyhalo_1.lens_cosmo),
                               'concentration_model_subhalos': ConcentrationDiemerJoyce(pyhalo_1.lens_cosmo),
                               'truncation_model_field_halos': TruncationRN(pyhalo_1.lens_cosmo),
                               'concentration_model_field_halos': ConcentrationDiemerJoyce(pyhalo_1.lens_cosmo),
                               'kwargs_density_profile': {}}
        realization_list_1 = pyhalo_1.render(population_model_list, mass_function_class_list,
                                             kwargs_mass_function_list,
                                             spatial_distribution_class_list,
                                             kwargs_spatial_distribution_list_1,
                                             geometry_1, mdef_subhalos, mdef_field_halos, kwargs_halo_model_1,
                                             nrealizations=1)
        nhalos_1 = len(realization_list_1[0].halos)


        kwargs_cosmo_2 = {'H0': 70.0,
                          'Ob0': 0.05,
                          'Om0': 0.8,
                          'sigma8': 1.0,
                          'flat': True,
                          'ns': 0.965,
                          'power_law': False}
        pyhalo_2 = pyHalo(0.5, 2.0, kwargs_cosmo_2)
        geometry_2 = Geometry(pyhalo_2.cosmology, pyhalo_2.zlens, pyhalo_2.zsource, cone_opening_angle,
                              'DOUBLE_CONE')
        kwargs_spatial_distribution_list_2 = [{'cone_opening_angle': 6.0, 'geometry': geometry_2}]
        kwargs_halo_model_2 = {'truncation_model_subhalos': TruncationRoche(pyhalo_2.lens_cosmo),
                             'concentration_model_subhalos': ConcentrationDiemerJoyce(pyhalo_2.astropy_cosmo),
                             'truncation_model_field_halos': TruncationRN(pyhalo_2.lens_cosmo),
                             'concentration_model_field_halos': ConcentrationDiemerJoyce(pyhalo_2.astropy_cosmo),
                             'kwargs_density_profile': {}}
        realization_list_2 = pyhalo_2.render(population_model_list, mass_function_class_list,
                                             kwargs_mass_function_list,
                                             spatial_distribution_class_list,
                                             kwargs_spatial_distribution_list_2,
                                             geometry_2, mdef_subhalos, mdef_field_halos, kwargs_halo_model_2,
                                             nrealizations=1)
        nhalos_2 = len(realization_list_2[0].halos)
        npt.assert_array_less(nhalos_1, nhalos_2)


if __name__ == '__main__':

    pytest.main()
