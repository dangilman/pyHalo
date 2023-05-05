import pytest
import numpy as np
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Cosmology.geometry import Geometry
import numpy.testing as npt
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Rendering.SpatialDistributions.nfw import ProjectedNFW
from pyHalo.Rendering.MassFunctions.mass_function_base import CDMPowerLaw
from pyHalo.Rendering.subhalos import Subhalos, normalization_sigmasub

class TestSubhalos(object):

    def setup_method(self):

        sigma_sub = 0.75
        kwargs_mass_function = {'log_mlow': 6.0, 'log_mhigh': 8.0, 'm_pivot': 1e8, 'power_law_index': -1.9,
                                'delta_power_law_index': 0.054, 'draw_poisson': False, 'log_m_host': 13.5,
                                'sigma_sub': sigma_sub}
        zlens = 0.8
        zsource = 1.5
        cosmo = Cosmology()
        geometry = Geometry(cosmo, zlens, zsource, 6.0, 'DOUBLE_CONE')
        lens_cosmo = LensCosmo(zlens, zsource, cosmo)
        mhost = 10 ** kwargs_mass_function['log_m_host']
        spatial_distribution_model = ProjectedNFW.from_Mhost(mhost, zlens, geometry.cone_opening_angle/2,
                                                                           0.05, lens_cosmo)
        mass_function_model = CDMPowerLaw
        self.model = Subhalos(mass_function_model, kwargs_mass_function, spatial_distribution_model,
                              geometry, lens_cosmo)
        self.kwargs_mass_function = kwargs_mass_function
        self._norm_sigma_sub = normalization_sigmasub(kwargs_mass_function['sigma_sub'],
                                                      10**kwargs_mass_function['log_m_host'], zlens,
                                                      lens_cosmo.cosmo.kpc_proper_per_asec(zlens),
                                                      geometry.cone_opening_angle,
                                                      kwargs_mass_function['power_law_index']+kwargs_mass_function['delta_power_law_index'],
                                                      kwargs_mass_function['m_pivot'])

    def test_mass_rendered(self):

        model = CDMPowerLaw(self.kwargs_mass_function['log_mlow'], self.kwargs_mass_function['log_mhigh'],
                              self.kwargs_mass_function['power_law_index']+self.kwargs_mass_function['delta_power_law_index'],
                              False, self._norm_sigma_sub)
        mtheory = model.first_moment
        m_rendered = np.sum(self.model.render()[0])
        npt.assert_array_less(abs(1-mtheory/m_rendered), 0.03)

    def test_rendering(self):

        m, x, y, r3d_kpc, z, subhalo_flag = self.model.render()
        npt.assert_equal(len(m), len(x))
        npt.assert_equal(len(x), len(y))
        npt.assert_equal(len(r3d_kpc), len(z))
        npt.assert_equal(len(subhalo_flag), len(x))
        for flag in subhalo_flag:
            npt.assert_equal(True, flag)
        for zi in z:
            npt.assert_equal(zi, 0.8)

    def test_sheets(self):
        kwargs_out, profile_names_out, redshifts = self.model.convergence_sheet_correction(kappa_scale=1.0,
                                                                                           log_mlow=6.0,
                                                                                           log_mhigh=10.0)
        npt.assert_equal(len(kwargs_out), len(profile_names_out))
        npt.assert_equal(len(profile_names_out), len(redshifts))

if __name__ == '__main__':
    pytest.main()

