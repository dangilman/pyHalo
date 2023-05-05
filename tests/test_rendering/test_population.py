from pyHalo.Rendering.halo_population import HaloPopulation
from pyHalo.Rendering.MassFunctions.mass_function_base import CDMPowerLaw
from pyHalo.Rendering.MassFunctions.density_peaks import ShethTormen
from pyHalo.Rendering.SpatialDistributions.uniform import LensConeUniform
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Cosmology.geometry import Geometry
from pyHalo.Rendering.SpatialDistributions.nfw import ProjectedNFW
from pyHalo.utilities import generate_lens_plane_redshifts
import numpy.testing as npt
import pytest

class TestPopulation(object):

    def setup_method(self):

        cosmo = Cosmology()
        lens_cosmo = LensCosmo(0.5, 1.5, cosmo)
        geometry = Geometry(cosmo, 0.5, 1.5, 6.0, 'DOUBLE_CONE')
        model_list = ['SUBHALOS', 'LINE_OF_SIGHT', 'LINE_OF_SIGHT_NOSHEET', 'TWO_HALO']
        mass_function_class_list = [CDMPowerLaw, ShethTormen, ShethTormen, ShethTormen]
        kwargs_subhalos = {'log_mlow': 6.0, 'log_mhigh': 9.0, 'm_pivot': 1e8, 'power_law_index': -1.9,
                                'delta_power_law_index': 0.054, 'draw_poisson': False, 'log_m_host': 13.5,
                                'sigma_sub': 0.05}
        kwargs_los = {'log_mlow': 6.0, 'log_mhigh': 10.0, 'm_pivot': 1e8,
                                'LOS_normalization': 1.0, 'delta_power_law_index': 0.0}
        kwargs_two_halo = {'log_mlow': 6.0, 'log_mhigh': 10.0, 'm_pivot': 1e8,
                                'LOS_normalization': 1.0, 'delta_power_law_index': 0.0, 'log_m_host': 13.0}
        kwargs_mass_function_list = [kwargs_subhalos, kwargs_los, kwargs_los, kwargs_two_halo]
        spatial_distribution_class_list = [ProjectedNFW, LensConeUniform, LensConeUniform, LensConeUniform]
        kwargs_subhalos_spatial = {'m_host': 10**13.0, 'zlens': 0.5, 'rmax2d_arcsec': 3.0, 'r_core_units_rs': 0.2,
                                   'lens_cosmo': lens_cosmo}
        kwargs_los_spatial = {'cone_opening_angle': 6.0, 'geometry': geometry}
        lens_plane_redshift_list, redshift_spacings = generate_lens_plane_redshifts(0.5, 1.5)
        kwargs_spatial_distribution = [kwargs_subhalos_spatial, kwargs_los_spatial, kwargs_los_spatial, kwargs_los_spatial]
        self.model = HaloPopulation(model_list, mass_function_class_list, kwargs_mass_function_list,
                 spatial_distribution_class_list, kwargs_spatial_distribution,
                 lens_cosmo, geometry,
                 lens_plane_redshift_list, redshift_spacings)

    def test_render(self):

        masses, x, y, r3d, redshifts, is_subhalo_flag = self.model.render()
        npt.assert_equal(len(masses), len(x))
        npt.assert_equal(len(masses), len(y))
        npt.assert_equal(len(masses), len(r3d))
        npt.assert_equal(len(masses), len(redshifts))
        npt.assert_equal(len(masses), len(is_subhalo_flag))

if __name__ == '__main__':
   pytest.main()
