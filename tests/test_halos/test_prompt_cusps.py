from pyHalo.single_realization import SingleHalo
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.concentration_models import preset_concentration_models
from pyHalo.realization_extensions import RealizationExtensions
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import numpy.testing as npt
from copy import deepcopy
import pytest


class TestPromptCusp(object):

    def setup_method(self):

        halo_mass = 10 ** 8
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

    def test_mass_conservation(self):

        ext = RealizationExtensions(deepcopy(self.single_halo_realization))
        mass_fraction = 0.35
        realization_with_PS = ext.add_prompt_cusps(mass_fraction)
        rs_kpc = self.single_halo_realization.halos[0].nfw_params[1]
        nfw_halo = realization_with_PS.halos[0]
        ps_halo = realization_with_PS.halos[1]

        r = np.linspace(0.001, nfw_halo.c, 10000) * rs_kpc
        rho_nfw = nfw_halo.density_profile_3d(r, scaling=1-mass_fraction)
        rho_ps = ps_halo.density_profile_3d(r)
        rho_total = rho_nfw + rho_ps
        rho_total_interp = interp1d(r, rho_total)

        mass_integrand_total = lambda x: 4 * np.pi * x ** 2 * rho_total_interp(x)
        total_mass = quad(mass_integrand_total, 0.001*rs_kpc, rs_kpc * nfw_halo.c)[0]
        npt.assert_almost_equal(-1 + total_mass/nfw_halo.mass, 0.0, 0.025)

    def test_rescaling_normalization(self):

        nfw_halo = self.single_halo_realization.halos[0]
        npt.assert_almost_equal(nfw_halo._rescale_norm, 1)

        ext = RealizationExtensions(deepcopy(self.single_halo_realization))
        mass_fraction = 0.2
        realization_with_PS = ext.add_prompt_cusps(mass_fraction)

        npt.assert_almost_equal(realization_with_PS.halos[0]._rescale_norm, 0.8, 2)


if __name__ == '__main__':
    pytest.main()
