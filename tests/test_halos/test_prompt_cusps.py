from pyHalo.single_realization import SingleHalo
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.concentration_models import preset_concentration_models
from pyHalo.realization_extensions import RealizationExtensions
import numpy as np
from scipy.integrate import quad
import numpy.testing as npt
from copy import deepcopy
import pytest


class TestPromptCusp(object):

    def setup_method(self):

        halo_mass = 10 ** 7.7
        self.nfw_mass = halo_mass
        x_arcsec = 0.
        y_arcsec = 0.0
        z_halo = 0.5

        zsource = 1.5
        # the source redshift is needed to specify the center of the rendering volume, something necessary for full
        # halo populations. The source redshift won't actually matter for single-halo models. However, if you want to do
        # multi-plane lensing computations with lenstronomy, you will need the source redshift
        zlens = 0.5  # possibly irrelevant for single-halo models, depending on whether the object is a subhalo
        mass_definition = 'NFW'
        subhalo_flag = False
        lens_cosmo = LensCosmo(zlens, zsource)
        astropy_class = lens_cosmo.cosmo

        model, kwargs_concentration_model = preset_concentration_models('DIEMERJOYCE19')
        kwargs_concentration_model['scatter'] = False
        kwargs_concentration_model['cosmo'] = astropy_class
        concentration_model = model(**kwargs_concentration_model)

        halo_truncation_model = None  # an NFW halo has no formal truncation radius, so we don't need to specify this
        kwargs_halo_model = {'truncation_model': halo_truncation_model,
                             'concentration_model': concentration_model,
                             'kwargs_density_profile': {}}
        self.single_halo_realization = SingleHalo(halo_mass, x_arcsec, y_arcsec, mass_definition, z_halo, zlens, zsource,
                                 subhalo_flag, kwargs_halo_model=kwargs_halo_model)

    def test_rescaling_normalization(self):

        ext = RealizationExtensions(deepcopy(self.single_halo_realization))
        a = 0.1
        b = 0.0
        c = 0.0
        realization_with_PS = ext.add_prompt_cusps(a,b,c)
        nfw_halo = realization_with_PS.halos[0]
        ps_halo = realization_with_PS.halos[1]
        _ = nfw_halo.lenstronomy_params
        _ = ps_halo.lenstronomy_params
        npt.assert_almost_equal(nfw_halo._rescale_norm, 0.9)
        npt.assert_almost_equal(ps_halo._rescale_norm, 1.0)

    def test_mass_conservation(self):

        ext = RealizationExtensions(deepcopy(self.single_halo_realization))
        a = 0.25
        b = 0.0
        c = 0.0
        realization_with_PS = ext.add_prompt_cusps(a, b, c)
        nfw_halo = realization_with_PS.halos[0]
        ps_halo = realization_with_PS.halos[1]
        rescale_norm_nfw = nfw_halo._rescale_norm
        rs = nfw_halo.nfw_params[1]
        r = np.logspace(-1.5, 0.7, 100) * rs
        integrand_total = lambda x: 4 * np.pi * x ** 2 * (rescale_norm_nfw * nfw_halo.density_profile_3d(x) +
                                                          ps_halo.density_profile_3d(x))
        integrand_ps = lambda x: 4 * np.pi * x ** 2 * ps_halo.density_profile_3d(x)

        # test that the total mass of the ps_halo is what it should be
        mass_target = nfw_halo.mass * a
        _, cusp_R, cusp_A = ps_halo.profile_args
        ps_halo_mass = 8 * np.pi / 3 * cusp_A * cusp_R ** 1.5
        npt.assert_almost_equal(np.log10(mass_target), np.log10(ps_halo_mass))
        npt.assert_almost_equal(np.log10(mass_target) / np.log10(quad(integrand_ps, 0, nfw_halo.c * rs))[0], 1, 2)


        total_mass = quad(integrand_total, 0, nfw_halo.c * rs)[0]
        npt.assert_array_less(abs(100*(total_mass / nfw_halo.mass - 1)), 1.6)

if __name__ == '__main__':
    pytest.main()
