import pytest
import numpy as np
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Cosmology.cosmology import Cosmology
import numpy.testing as npt
from pyHalo.Halos.concentration import *
from astropy.cosmology import FlatLambdaCDM

class TestConcentration(object):

    def setup(self):

        astropy = FlatLambdaCDM(70.0, 0.3)
        cosmo = Cosmology(astropy_instance=astropy)
        self.cosmo = cosmo
        self.lenscosmo = LensCosmo(0.5, 1.5, cosmo)

    def test_concentration_diemer_joyce(self):

        m = 10 ** 8
        z = 0.5
        c_true = 14.246936385951503
        scatter = False
        scatter_amplitude_dex = 0.2
        concentration_model = ConcentrationDiemerJoyce(self.lenscosmo)
        c = concentration_model.nfw_concentration(m, z, scatter, scatter_amplitude_dex)
        npt.assert_almost_equal(c_true, c)

        scatter = True
        c = concentration_model.nfw_concentration(m, z, scatter, scatter_amplitude_dex)
        npt.assert_equal(c != c_true, True)

    def test_concentration_peak_height(self):

        c_true = 14.24
        m = 10 ** 8
        z = 0.5
        c0 = 21.49
        zeta = -0.5
        beta = 0.8
        scatter = False
        scatter_amplitude_dex = 0.2
        concentration_model = ConcentrationPeakHeight(self.lenscosmo, c0, zeta, beta)
        c1 = concentration_model.nfw_concentration(m, z, scatter, scatter_amplitude_dex)
        npt.assert_almost_equal(c1, c_true, 2)

        scatter = True
        c = concentration_model.nfw_concentration(m, z, scatter, scatter_amplitude_dex)
        npt.assert_equal(c != c_true, True)
        args = (self.lenscosmo, c0, 0.5, beta)
        npt.assert_raises(Exception, ConcentrationPeakHeight, args)

    def test_concentration_wdm_polynomial(self):

        m = 10 ** 8
        z = 0.5
        c0 = 21.49
        zeta = -0.5
        beta = 0.8
        kwargs_cdm = {'c0': c0, 'zeta': zeta, 'beta': beta}
        concentration_cdm = ConcentrationPeakHeight
        concentration_cdm_peak_height = concentration_cdm(self.lenscosmo, **kwargs_cdm)
        log_mc = 8.2
        c_scale = 60.0
        c_power = -0.17
        c_power_inner = 1.5

        mc_suppression_redshift_evolution = False
        concentration_model = ConcentrationWDMPolynomial(self.lenscosmo, concentration_cdm,
                                                         log_mc, c_scale, c_power, c_power_inner, mc_suppression_redshift_evolution,
                                                         kwargs_cdm)
        scatter = False
        scatter_amplitude_dex = 0.2
        c_wdm = concentration_model.nfw_concentration(m, z, scatter, scatter_amplitude_dex)
        c_cdm = concentration_cdm_peak_height.nfw_concentration(m, z, scatter, scatter_amplitude_dex)
        suppression = (1 + c_scale * (10**log_mc / m)**c_power_inner)**c_power
        npt.assert_almost_equal(c_wdm, c_cdm * suppression)

        mc_suppression_redshift_evolution = True
        concentration_model = ConcentrationWDMPolynomial(self.lenscosmo, concentration_cdm,
                                                         log_mc, c_scale, c_power, c_power_inner,
                                                         mc_suppression_redshift_evolution,
                                                         kwargs_cdm)
        scatter = False
        scatter_amplitude_dex = 0.2
        c_wdm = concentration_model.nfw_concentration(m, z, scatter, scatter_amplitude_dex)
        c_cdm = concentration_cdm_peak_height.nfw_concentration(m, z, scatter, scatter_amplitude_dex)
        suppression *= (1 + z) ** (0.026 * z - 0.04)
        npt.assert_almost_equal(c_wdm, c_cdm * suppression)

    def test_concentration_wdm_hyperbolic(self):

        m = 10 ** 8
        z = 0.5
        kwargs_cdm = {}
        concentration_cdm = ConcentrationDiemerJoyce
        concentration_cdm_diemer_joyce = concentration_cdm(self.lenscosmo, **kwargs_cdm)
        log_mc = 8.2
        a = 0.2
        b = 0.5
        concentration_model = ConcentrationWDMHyperbolic(self.lenscosmo, concentration_cdm, log_mc, a, b, kwargs_cdm)
        scatter = False
        scatter_amplitude_dex = 0.2
        c_wdm = concentration_model.nfw_concentration(m, z, scatter, scatter_amplitude_dex)
        c_cdm = concentration_cdm_diemer_joyce.nfw_concentration(m, z, scatter, scatter_amplitude_dex)
        suppression = 0.5 * (1 + np.tanh((np.log10(m/10**log_mc) - a) / (2 * b)))
        npt.assert_almost_equal(c_wdm, c_cdm * suppression)

if __name__ == '__main__':
    pytest.main()
