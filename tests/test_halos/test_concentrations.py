import pytest
import numpy as np
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Cosmology.cosmology import Cosmology
import numpy.testing as npt
from pyHalo.Halos.concentration import *
from astropy.cosmology import FlatLambdaCDM

class TestConcentration(object):

    def setup_method(self):

        astropy = FlatLambdaCDM(70.0, 0.3)
        cosmo = Cosmology(astropy_instance=astropy)
        self.astropy = cosmo.astropy

    def test_concentration_diemer_joyce(self):

        m = 10 ** 8
        z = 0.5
        c_true = 14.246936385951503
        scatter = False
        scatter_amplitude_dex = 0.2
        concentration_model = ConcentrationDiemerJoyce(self.astropy, scatter)
        c = concentration_model.nfw_concentration(m, z)
        npt.assert_almost_equal(c_true, c)

        scatter = True
        concentration_model = ConcentrationDiemerJoyce(self.astropy, scatter)
        c = concentration_model.nfw_concentration(m, z)
        npt.assert_equal(c != c_true, True)

    def test_concentration_peak_height(self):

        m = 10 ** 8
        concentration_model_1 = ConcentrationPeakHeight(self.astropy, 16.0, -0.2, 0.8, False)
        concentration_model_2 = ConcentrationPeakHeight(self.astropy, 16.0, -0.2, 4.0, False)
        concentration_model_3 = ConcentrationPeakHeight(self.astropy, 2.5 * 16.0, -0.2, 4.0, False)
        c1 = concentration_model_1.nfw_concentration(m, 0.5)
        c2 = concentration_model_2.nfw_concentration(m, 0.5)
        c3 = concentration_model_3.nfw_concentration(m, 0.5)
        npt.assert_equal(c1, c2)
        npt.assert_almost_equal(c1, c3 / 2.5, 6)

    def test_concentration_wdm_polynomial(self):

        m = 10 ** 8
        z = 0.5
        c0 = 21.49
        zeta = -0.5
        beta = 0.8
        kwargs_cdm = {'c0': c0, 'zeta': zeta, 'beta': beta, 'scatter': False}
        concentration_cdm = ConcentrationPeakHeight
        concentration_cdm_peak_height = concentration_cdm(self.astropy, **kwargs_cdm)
        log_mc = 8.2
        c_scale = 60.0
        c_power = -0.17
        c_power_inner = 1.5

        mc_suppression_redshift_evolution = False
        concentration_model = ConcentrationWDMPolynomial(self.astropy, concentration_cdm,
                                                         log_mc, c_scale, c_power, c_power_inner, mc_suppression_redshift_evolution,
                                                         scatter=False, kwargs_cdm=kwargs_cdm)

        c_wdm = concentration_model.nfw_concentration(m, z)
        c_cdm = concentration_cdm_peak_height.nfw_concentration(m, z)
        suppression = (1 + c_scale * (10**log_mc / m)**c_power_inner)**c_power
        npt.assert_almost_equal(c_wdm, c_cdm * suppression)

        mc_suppression_redshift_evolution = True
        concentration_model = ConcentrationWDMPolynomial(self.astropy, concentration_cdm,
                                                         log_mc, c_scale, c_power, c_power_inner,
                                                         mc_suppression_redshift_evolution,
                                                         scatter=False, kwargs_cdm=kwargs_cdm)

        c_wdm = concentration_model.nfw_concentration(m, z)
        c_cdm = concentration_cdm_peak_height.nfw_concentration(m, z)
        suppression *= (1 + z) ** (0.026 * z - 0.04)
        npt.assert_almost_equal(c_wdm, c_cdm * suppression)

    def test_concentration_wdm_hyperbolic(self):

        m = 10 ** 8
        z = 0.5
        kwargs_cdm = {'scatter': False}
        concentration_cdm = ConcentrationDiemerJoyce
        concentration_cdm_diemer_joyce = concentration_cdm(self.astropy, **kwargs_cdm)
        log_mc = 8.2
        a = 0.2
        b = 0.5
        concentration_model = ConcentrationWDMHyperbolic(self.astropy, concentration_cdm, log_mc, a, b,
                                                         scatter=False,kwargs_cdm=kwargs_cdm)
        c_wdm = concentration_model.nfw_concentration(m, z)
        c_cdm = concentration_cdm_diemer_joyce.nfw_concentration(m, z)
        suppression = 0.5 * (1 + np.tanh((np.log10(m/10**log_mc) - a) / (2 * b)))
        npt.assert_almost_equal(c_wdm, c_cdm * suppression)

if __name__ == '__main__':
    pytest.main()
