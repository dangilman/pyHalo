import pytest
import numpy as np
from pyHalo.utilities import generate_lens_plane_redshifts
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Cosmology.geometry import Geometry
import numpy.testing as npt
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Rendering.SpatialDistributions.uniform import LensConeUniform
from pyHalo.Rendering.MassFunctions.density_peaks import ShethTormen
from pyHalo.Rendering.line_of_sight import LineOfSightNoSheet, LineOfSight

class TestLineOfSightNoSheet(object):

    def setup_method(self):

        mass_function_model = ShethTormen
        kwargs_mass_function = {'log_mlow': 6.0, 'log_mhigh': 10.0, 'm_pivot': 1e8,
                                'LOS_normalization': 1.0, 'delta_power_law_index': 0.0}
        zlens = 0.5
        zsource = 1.5
        cosmo = Cosmology()
        geometry = Geometry(cosmo, zlens, zsource, 6.0, 'DOUBLE_CONE')
        lens_cosmo = LensCosmo(zlens, zsource, cosmo)
        lens_plane_redshifts, delta_z_list = generate_lens_plane_redshifts(zlens, zsource)
        spatial_distribution_model = LensConeUniform(geometry.cone_opening_angle, geometry)
        self.model = LineOfSightNoSheet(mass_function_model, kwargs_mass_function, spatial_distribution_model,
                 geometry, lens_cosmo, lens_plane_redshifts, delta_z_list)
        self._is_subhalo = False

    def test_render(self):

        masses, x, y, r3d, redshifts, subhalo_flag = self.model.render()
        npt.assert_equal(len(masses), len(x))
        npt.assert_equal(len(masses), len(y))
        npt.assert_equal(len(redshifts), len(masses))
        for flag in subhalo_flag:
            npt.assert_equal(flag, self._is_subhalo)

class TestLineOfSight(object):

    def setup_method(self):

        mass_function_model = ShethTormen
        kwargs_mass_function = {'log_mlow': 6.0, 'log_mhigh': 10.0, 'm_pivot': 1e8,
                                'LOS_normalization': 1.0, 'delta_power_law_index': 0.0}
        zlens = 0.5
        zsource = 1.5
        cosmo = Cosmology()
        geometry = Geometry(cosmo, zlens, zsource, 6.0, 'DOUBLE_CONE')
        lens_cosmo = LensCosmo(zlens, zsource, cosmo)
        lens_plane_redshifts, delta_z_list = generate_lens_plane_redshifts(zlens, zsource)
        spatial_distribution_model = LensConeUniform(geometry.cone_opening_angle, geometry)
        self.model = LineOfSight(mass_function_model, kwargs_mass_function, spatial_distribution_model,
                 geometry, lens_cosmo, lens_plane_redshifts, delta_z_list)
        self._is_subhalo = False

    def test_render(self):

        masses, x, y, r3d, redshifts, subhalo_flag = self.model.render()
        npt.assert_equal(len(masses), len(x))
        npt.assert_equal(len(masses), len(y))
        npt.assert_equal(len(redshifts), len(masses))
        for flag in subhalo_flag:
            npt.assert_equal(flag, self._is_subhalo)

    def test_sheets(self):
        kwargs_out, profile_names_out, redshifts = self.model.convergence_sheet_correction(kappa_scale=1.0,
                                                                                           log_mlow=6.0,
                                                                                           log_mhigh=10.0)
        npt.assert_equal(len(kwargs_out), len(profile_names_out))
        npt.assert_equal(len(profile_names_out), len(redshifts))

if __name__ == '__main__':
    pytest.main()

