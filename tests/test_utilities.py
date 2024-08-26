from lenstronomy.LensModel.lens_model import LensModel
import numpy.testing as npt
from scipy.integrate import simpson as simps
import numpy as np
from pyHalo.utilities import interpolate_ray_paths, de_broglie_wavelength, delta_sigma, ITSampling, \
    inverse_transform_sampling, delta_kappa, nfw_velocity_dispersion
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.utilities import mask_annular
import pytest

class TestUtilities(object):

    def test_interp_ray_paths(self):

        cosmo = Cosmology()
        x = [1.4, -1.]
        y = [-0.4, 0.2]
        lens_model = LensModel(['SIS'], z_source=2., lens_redshift_list=[0.5], multi_plane=True)
        kwargs_lens = [{'theta_E': 1., 'center_x': 0., 'center_y': 0.}]

        interpx, interpy = interpolate_ray_paths(x, y, lens_model, kwargs_lens, 2.,
                                                             terminate_at_source=False, source_x=None, source_y=None,
                                                             evaluate_at_mean=False)
        zarray = np.linspace(0., 2., 100)
        dc = [cosmo.D_C_transverse(zi) for zi in zarray]
        interpx_mean, interpy_mean = interpolate_ray_paths(x, y, lens_model, kwargs_lens, 2.,
                                                                       terminate_at_source=False, source_x=None,
                                                                       source_y=None,
                                                                       evaluate_at_mean=True, cosmo=cosmo)
        for dci in dc:
            x, y = 0.5 * (interpx[0](dci) + interpx[1](dci)), 0.5 * (interpy[0](dci) + interpy[1](dci))
            npt.assert_almost_equal(x, interpx_mean[0](dci))
            npt.assert_almost_equal(y, interpy_mean[0](dci))

        x = [1.4, -1.]
        y = [-0.4, 0.2]
        source_x, source_y = 0.2, 0.5
        interpx, interpy = interpolate_ray_paths(x, y, lens_model, kwargs_lens, 2.,
                                                             terminate_at_source=True, source_x=source_x,
                                                             source_y=source_y,
                                                             evaluate_at_mean=True)
        npt.assert_almost_equal(interpx[0](dc[-1]), source_x)
        npt.assert_almost_equal(interpy[0](dc[-1]), source_y)

    def test_uldm_functions(self):

        log10_m_uldm=-22 # log(eV)
        v=200 # km/s
        z_lens,z_source=0.5,1.5
        m=1e13 #M_solar
        rein=6. #kpc
        lens_cosmo = LensCosmo(z_lens, z_source)
        lambda_dB = de_broglie_wavelength(log10_m_uldm,v)
        npt.assert_almost_equal(lambda_dB,0.6)
        delta_s = delta_sigma(m,z_lens,rein,lambda_dB)
        npt.assert_almost_equal(delta_s/80837585.696, 1, 2)
        dk = delta_kappa(z_lens, z_source, 10 ** 13.0, rein, lambda_dB)
        sigma_crit = lens_cosmo.get_sigma_crit_lensing(z_lens, z_source) * (1e-3) ** 2
        npt.assert_almost_equal(delta_s, dk * sigma_crit)

    def test_nfw_velocity_dispersion(self):

        lens_comso = LensCosmo(0.5, 1.5)
        m = 10**8
        z = 0.5
        c = 15.0
        rhos, rs, _ = lens_comso.NFW_params_physical(m, c, z)
        vdis1 = nfw_velocity_dispersion(rhos, rs, c)
        npt.assert_almost_equal(vdis1/6.34, 1.0, 2)

    def test_inverse_transform_sampling(self):

        mu = 2.0
        sigma = 1.5
        func = lambda x: np.exp(-0.5 * (x - mu) ** 2 / sigma**2)
        x = np.linspace(mu - 5*sigma, mu + 5*sigma, 1000)
        x_samples = inverse_transform_sampling(x, func, (), 100000)
        npt.assert_almost_equal(np.mean(x_samples)/mu, 1.0, 2)
        npt.assert_almost_equal(np.std(x_samples)/sigma, 1.0, 2)

    def test_ITSampling(self):

        mu = 2.14
        sigma = 0.35
        x = np.random.normal(mu, sigma, 250000)
        sampler = ITSampling.from_samples(x)
        x_samples = sampler(100000)
        npt.assert_almost_equal(np.mean(x_samples)/mu, 1.0, 2)
        npt.assert_almost_equal(np.std(x_samples)/sigma, 1.0, 2)

    def test_mask_annular(self):
        npix = 1000
        window_size = 4
        delta_pix = window_size/npix
        _R = np.linspace(-window_size/2, window_size/2, npix)
        XX, YY = np.meshgrid(_R, _R)
        r_min, r_max = 0.5, 1.5
    
        no_mask = mask_annular(0, 0, XX, YY, r_min = 0, r_max = None)
        mask_1 = mask_annular(0, 0, XX, YY, r_min, r_max = None)
        mask_2 = mask_annular(0, 0, XX, YY, r_min, r_max)
    
        area_no_mask = simps(simps(no_mask, _R, axis=0), _R, axis=-1)
        area_mask_1 = simps(simps(mask_1, _R, axis=0), _R, axis=-1)
        area_mask_2 = simps(simps(mask_2, _R, axis=0), _R, axis=-1)
    
        area_no_mask_real = window_size**2
        area_mask_1_real = window_size**2 - np.pi*r_min**2
        area_mask_2_real = np.pi*(r_max**2 - r_min**2)
    
        npt.assert_array_almost_equal(area_no_mask_real,area_no_mask, decimal=3)
        npt.assert_array_almost_equal(area_mask_1_real,area_mask_1, decimal=3)
        npt.assert_array_almost_equal(area_mask_2_real,area_mask_2, decimal=3)

if __name__ == '__main__':

    pytest.main()
