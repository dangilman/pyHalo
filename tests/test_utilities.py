from lenstronomy.LensModel.lens_model import LensModel
import numpy.testing as npt
import numpy as np
from pyHalo.utilities import interpolate_ray_paths, de_broglie_wavelength, delta_sigma, ITSampling, \
    inverse_transform_sampling, inverse_transform_sampling_from_cdf
from pyHalo.Cosmology.cosmology import Cosmology
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

        lambda_dB = de_broglie_wavelength(log10_m_uldm,v)
        npt.assert_almost_equal(lambda_dB,0.6)
        delta_kappa = delta_sigma(m,z_lens,rein,lambda_dB)
        npt.assert_almost_equal(delta_kappa/80837585.696, 1, 2)

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


if __name__ == '__main__':

    pytest.main()
