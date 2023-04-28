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

        self.pyhalo = pyHalo(0.5, 2.)

    def test_create_realizations(self):

        cone_opening_angle = 6.0
        geometry = Geometry(self.pyhalo.cosmology, self.pyhalo.zlens, self.pyhalo.zsource, cone_opening_angle, 'DOUBLE_CONE')
        population_model_list = ['SUBHALOS']
        mass_function_class_list = [CDMPowerLaw, ShethTormen, ShethTormen, ShethTormen]
        kwargs_subhalos = {'log_mlow': 6.0, 'log_mhigh': 10.0, 'm_pivot': 10**8, 'power_law_index': -1.9,
                           'delta_power_law_index': 0.0, 'draw_poisson': False, 'log_m_host': 13.3,
                           'sigma_sub': 0.05}
        kwargs_los = {'log_mlow': 6.0, 'log_mhigh': 10.0, 'm_pivot': 1e8,
                      'LOS_normalization': 1.0, 'delta_power_law_index': 0.0}
        kwargs_two_halo = {'log_mlow': 6.0, 'log_mhigh': 10.0, 'm_pivot': 1e8,
                           'LOS_normalization': 1.0, 'delta_power_law_index': 0.0, 'log_m_host': 13.0}
        kwargs_mass_function_list = [kwargs_subhalos, kwargs_los, kwargs_two_halo]
        spatial_distribution_class_list = [ProjectedNFW.from_Mhost, LensConeUniform, LensConeUniform]
        kwargs_subhalos_spatial = {'m_host': 10 ** 13.0, 'zlens': 0.5, 'rmax2d_arcsec': 3.0, 'r_core_units_rs': 0.2,
                                   'lens_cosmo': self.pyhalo.lens_cosmo}
        kwargs_los_spatial = {'cone_opening_angle': 6.0, 'geometry': geometry}
        kwargs_spatial_distribution_list = [kwargs_subhalos_spatial, kwargs_los_spatial,
                                       kwargs_los_spatial]
        mdef_subhalos = 'TNFW'
        mdef_field_halos = 'TNFW'
        kwargs_halo_model = {'truncation_model_subhalos': TruncationRoche(self.pyhalo.lens_cosmo),
                             'concentration_model_subhalos': ConcentrationDiemerJoyce(self.pyhalo.lens_cosmo),
                             'truncation_model_field_halos': TruncationRN(self.pyhalo.lens_cosmo),
                             'concentration_model_field_halos': ConcentrationDiemerJoyce(self.pyhalo.lens_cosmo),
                             'args': {}}
        realization_list = self.pyhalo.render(population_model_list, mass_function_class_list, kwargs_mass_function_list,
                   spatial_distribution_class_list, kwargs_spatial_distribution_list,
               geometry, mdef_subhalos, mdef_field_halos, kwargs_halo_model, nrealizations=1)
        #npt.assert_equal(len(realization_list), 2)
        print(len(realization_list[0].halos))

    def test_cosmology_setup_from_default(self):

        astropy = self.pyhalo.astropy_cosmo
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
#
t = TestpyHalo()
t.setup_method()
t.test_create_realizations()
# if __name__ == '__main__':
#
#     pytest.main()
