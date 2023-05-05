import pytest
import numpy as np
import numpy.testing as npt
from pyHalo.Rendering.two_halo import TwoHaloContribution, two_halo_enhancement_factor
from pyHalo.Rendering.MassFunctions.density_peaks import ShethTormen
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Cosmology.geometry import Geometry
from pyHalo.utilities import generate_lens_plane_redshifts
from pyHalo.Rendering.SpatialDistributions.uniform import LensConeUniform


class TestTwoHalo(object):

    def setup_method(self):

        self.zlens = 0.5
        self.zsource = 1.5
        self.cosmo = Cosmology()
        self.lens_cosmo = LensCosmo(self.zlens, self.zsource, self.cosmo)

        mass_function_model = ShethTormen
        kwargs_mass_function = {'log_mlow': 6.0, 'log_mhigh': 10.0, 'm_pivot': 1e8,
                                'LOS_normalization': 1.0, 'delta_power_law_index': 0.0, 'log_m_host': 13.0}

        geometry = Geometry(self.cosmo, self.zlens, self.zsource, 6.0, 'DOUBLE_CONE')
        lens_cosmo = LensCosmo(self.zlens, self.zsource, self.cosmo)
        lens_plane_redshifts, delta_z_list = generate_lens_plane_redshifts(self.zlens, self.zsource)
        spatial_distribution_model = LensConeUniform(geometry.cone_opening_angle, geometry)
        self.model = TwoHaloContribution(mass_function_model, kwargs_mass_function,
                                                       spatial_distribution_model, geometry, lens_cosmo,
                                                       lens_plane_redshifts, delta_z_list)

    def test_boost(self):

        z_step = 0.02
        mhost = 10**13
        boost = two_halo_enhancement_factor(self.zlens, z_step, self.lens_cosmo, mhost)
        npt.assert_array_less(1.1, boost)

        z_step = 0.2
        mhost = 10 ** 13
        boost = two_halo_enhancement_factor(self.zlens, z_step, self.lens_cosmo, mhost)
        # as the redshift increment increases, the relative contribution from the two halo term diminishes
        npt.assert_array_less(boost, 1.1)

    def test_render(self):

        masses, x, y, r3d, redshifts, subhalo_flag = self.model.render()
        npt.assert_equal(len(masses), len(x))
        npt.assert_equal(len(masses), len(y))
        npt.assert_equal(len(masses), len(r3d))
        npt.assert_equal(len(masses), len(redshifts))
        npt.assert_equal(len(masses), len(subhalo_flag))
        for flag in subhalo_flag:
            npt.assert_equal(False, flag)
        npt.assert_equal(redshifts, self.zlens)


if __name__ == '__main__':
   pytest.main()
