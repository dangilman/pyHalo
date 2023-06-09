import pytest
import numpy as np
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Cosmology.geometry import Geometry
import numpy.testing as npt
from pyHalo.Rendering.MassFunctions.density_peaks import ShethTormen, ShethTormenTurnover, ShethTormenMixedWDM
from pyHalo.Rendering.MassFunctions.mass_function_base import CDMPowerLaw, WDMPowerLaw, MixedWDMPowerLaw
from colossus.lss.mass_function import massFunction

class TestShethTormen(object):

    def setup_method(self):

        z = 0.5
        delta_z = 0.02
        opening_angle = 16.0
        zlens = 0.5
        zsource = 1.5
        cosmo = Cosmology()
        _ = cosmo.colossus
        geometry_class = Geometry(cosmo, zlens, zsource, opening_angle, 'DOUBLE_CONE')
        delta_power_law_index = -0.1
        kwargs_model = {}
        kwargs_model['m_pivot'] = 10.0**8
        kwargs_model['log_mlow'] = 6.0
        kwargs_model['log_mhigh'] = 10.0
        kwargs_model['LOS_normalization'] = 1.0
        kwargs_model['draw_poisson'] = False
        kwargs_model['delta_power_law_index'] = delta_power_law_index

        h = geometry_class.cosmo.h
        # To M_sun / h units
        m = np.logspace(kwargs_model['log_mlow'], kwargs_model['log_mhigh'], 20)
        m_h = m * h

        dndlogM = massFunction(m_h, z, q_out='dndlnM', model='sheth99')
        dndM_comoving_h = dndlogM / m
        # three factors of h for the (Mpc/h)^-3 to Mpc^-3 conversion
        dndM_comoving = dndM_comoving_h * h ** 3
        coeffs = np.polyfit(np.log10(m / kwargs_model['m_pivot']), np.log10(dndM_comoving), 1)
        plaw_index = coeffs[0] + delta_power_law_index
        norm_dv = 10 ** coeffs[1]
        volume_element_comoving = geometry_class.volume_element_comoving(z, delta_z)
        normalization = 1.0 * norm_dv * volume_element_comoving
        normalization *= 1/(kwargs_model['m_pivot']**plaw_index)

        draw_poisson = False
        self.mass_function = CDMPowerLaw(kwargs_model['log_mlow'], kwargs_model['log_mhigh'], plaw_index,
                                         draw_poisson, normalization)
        draw_poisson = True
        self.mass_function_poisson = CDMPowerLaw(kwargs_model['log_mlow'], kwargs_model['log_mhigh'], plaw_index,
                                         draw_poisson, normalization)

        self._kwargs_from_redshift = {'z': z, 'delta_z': delta_z, 'geometry_class': geometry_class,
                                      'kwargs_model': kwargs_model}

    def test_draw_poisson(self):

        m = self.mass_function.draw()
        m_poisson = self.mass_function_poisson.draw()
        n1 = len(m)
        n2 = len(m_poisson)
        npt.assert_equal(True, n1!=n2)

    def test_from_redshift(self):
        mfunc = ShethTormen.from_redshift(**self._kwargs_from_redshift)
        m = mfunc.draw()
        npt.assert_almost_equal(len(m) / len(self.mass_function.draw()), 1, 3)

class TestShethTormenTurnover(object):

    def setup_method(self):

        z = 0.5
        delta_z = 0.02
        opening_angle = 16.0
        zlens = 0.5
        zsource = 1.5
        cosmo = Cosmology()
        geometry_class = Geometry(cosmo, zlens, zsource, opening_angle, 'DOUBLE_CONE')
        colossus_cosmology = cosmo.colossus
        np.random.seed(100)
        kwargs_model = {}
        delta_power_law_index = 0.0
        kwargs_model['m_pivot'] = 10.0 ** 8
        kwargs_model['log_mlow'] = 6.0
        kwargs_model['log_mhigh'] = 10.0
        kwargs_model['draw_poisson'] = False
        kwargs_model['LOS_normalization'] = 1.0
        kwargs_model['delta_power_law_index'] = delta_power_law_index
        kwargs_model['log_mc'] = 7.0
        kwargs_model['a_wdm'] = 1.0
        kwargs_model['b_wdm'] = 1.0
        kwargs_model['c_wdm'] = -2.5
        h = geometry_class.cosmo.h
        # To M_sun / h units
        m = np.logspace(kwargs_model['log_mlow'], kwargs_model['log_mhigh'], 20)
        m_h = m * h
        dndlogM = massFunction(m_h, z, q_out='dndlnM', model='sheth99')
        dndM_comoving_h = dndlogM / m
        # three factors of h for the (Mpc/h)^-3 to Mpc^-3 conversion
        dndM_comoving = dndM_comoving_h * h ** 3
        coeffs = np.polyfit(np.log10(m / kwargs_model['m_pivot']), np.log10(dndM_comoving), 1)

        plaw_index = coeffs[0] + delta_power_law_index
        norm_dv = 10 ** coeffs[1]
        volume_element_comoving = geometry_class.volume_element_comoving(z, delta_z)
        normalization = 1.0 * norm_dv * volume_element_comoving /(kwargs_model['m_pivot']**plaw_index)

        draw_poisson = False
        self.mass_function = WDMPowerLaw(kwargs_model['log_mlow'], kwargs_model['log_mhigh'], plaw_index,
                                         draw_poisson, normalization, kwargs_model['log_mc'], kwargs_model['a_wdm'],
                                         kwargs_model['b_wdm'], kwargs_model['c_wdm'])
        draw_poisson = True
        self.mass_function_poisson = WDMPowerLaw(kwargs_model['log_mlow'], kwargs_model['log_mhigh'], plaw_index,
                                         draw_poisson, normalization, kwargs_model['log_mc'], kwargs_model['a_wdm'],
                                         kwargs_model['b_wdm'], kwargs_model['c_wdm'])

        self._kwargs_from_redshift = {'z': z, 'delta_z': delta_z, 'geometry_class': geometry_class,
                                      'kwargs_model': kwargs_model}

    def test_draw_poisson(self):

        m = self.mass_function.draw()
        m_poisson = self.mass_function_poisson.draw()
        npt.assert_equal(True, len(m)!=len(m_poisson))

    def test_from_redshift(self):
        mfunc = ShethTormenTurnover.from_redshift(**self._kwargs_from_redshift)
        m = mfunc.draw()
        n1 = len(m)
        n2 = len(self.mass_function.draw())
        npt.assert_equal(True, n1==n2)

class TestShethTormenMixedDM(object):

    def setup_method(self):

        z = 0.5
        delta_z = 0.02
        opening_angle = 8.0
        zlens = 0.5
        zsource = 1.5
        cosmo = Cosmology()
        geometry_class = Geometry(cosmo, zlens, zsource, opening_angle, 'DOUBLE_CONE')
        _ = cosmo.colossus
        #np.random.seed(101)
        kwargs_model = {}
        delta_power_law_index = -0.1
        kwargs_model['m_pivot'] = 10.0 ** 8
        kwargs_model['log_mlow'] = 6.0
        kwargs_model['log_mhigh'] = 10.0
        kwargs_model['draw_poisson'] = False
        kwargs_model['LOS_normalization'] = 1.0
        kwargs_model['delta_power_law_index'] = delta_power_law_index
        kwargs_model['log_mc'] = 7.0
        kwargs_model['a_wdm'] = 1.0
        kwargs_model['b_wdm'] = 1.0
        kwargs_model['c_wdm'] = -1.2
        kwargs_model['mixed_DM_frac'] = 0.6
        h = geometry_class.cosmo.h
        # To M_sun / h units
        m = np.logspace(kwargs_model['log_mlow'], kwargs_model['log_mhigh'], 20)
        m_h = m * h
        dndlogM = massFunction(m_h, z, q_out='dndlnM', model='sheth99')
        dndM_comoving_h = dndlogM / m
        # three factors of h for the (Mpc/h)^-3 to Mpc^-3 conversion
        dndM_comoving = dndM_comoving_h * h ** 3
        coeffs = np.polyfit(np.log10(m / kwargs_model['m_pivot']), np.log10(dndM_comoving), 1)
        plaw_index = coeffs[0] + delta_power_law_index
        norm_dv = 10 ** coeffs[1]
        volume_element_comoving = geometry_class.volume_element_comoving(z, delta_z)
        normalization = 1.0 * norm_dv * volume_element_comoving /(kwargs_model['m_pivot']**plaw_index)
        draw_poisson = False

        self.mass_function = MixedWDMPowerLaw(kwargs_model['log_mlow'], kwargs_model['log_mhigh'], plaw_index,
                                         draw_poisson, normalization, kwargs_model['log_mc'], kwargs_model['a_wdm'],
                                         kwargs_model['b_wdm'], kwargs_model['c_wdm'], kwargs_model['mixed_DM_frac'])
        draw_poisson = True
        self.mass_function_poisson = MixedWDMPowerLaw(kwargs_model['log_mlow'], kwargs_model['log_mhigh'], plaw_index,
                                         draw_poisson, normalization, kwargs_model['log_mc'], kwargs_model['a_wdm'],
                                         kwargs_model['b_wdm'], kwargs_model['c_wdm'], kwargs_model['mixed_DM_frac'])

        self._kwargs_from_redshift = {'z': z, 'delta_z': delta_z, 'geometry_class': geometry_class,
                                      'kwargs_model': kwargs_model}

    def test_draw_poisson(self):

        m = self.mass_function.draw()
        m_poisson = self.mass_function_poisson.draw()
        n1 = len(m)
        n2 = len(m_poisson)
        npt.assert_equal(True, n1!=n2)

    def test_from_redshift(self):
        n_iter = 250
        n1 = 0
        n2 = 0
        for i in range(0, n_iter):
            mfunc = ShethTormenMixedWDM.from_redshift(**self._kwargs_from_redshift)
            m = mfunc.draw()
            n1 += len(m)
            n2 += len(self.mass_function.draw())
        n1 /= n_iter
        n2 /= n_iter
        npt.assert_almost_equal(n1/n2, 1, 2)

class TestSampling(object):

    def setup_method(self):

        z = 0.5
        delta_z = 0.02
        opening_angle = 16.0
        zlens = 0.5
        zsource = 1.5
        cosmo = Cosmology()
        _ = cosmo.colossus
        geometry_class = Geometry(cosmo, zlens, zsource, opening_angle, 'DOUBLE_CONE')
        kwargs_model = {}
        kwargs_model['m_pivot'] = 10.0**8
        kwargs_model['log_mlow'] = 6.0
        kwargs_model['log_mhigh'] = 10.0
        kwargs_model['draw_poisson'] = False
        delta_power_law_index = -0.1
        h = geometry_class.cosmo.h
        # To M_sun / h units
        m = np.logspace(kwargs_model['log_mlow'], kwargs_model['log_mhigh'], 20)
        m_h = m * h

        dndlogM = massFunction(m_h, z, q_out='dndlnM', model='sheth99')
        dndM_comoving_h = dndlogM / m
        # three factors of h for the (Mpc/h)^-3 to Mpc^-3 conversion
        dndM_comoving = dndM_comoving_h * h ** 3
        coeffs = np.polyfit(np.log10(m / kwargs_model['m_pivot']), np.log10(dndM_comoving), 1)
        plaw_index = coeffs[0] + delta_power_law_index
        norm_dv = 10 ** coeffs[1]
        volume_element_comoving = geometry_class.volume_element_comoving(z, delta_z)
        normalization = 1.0 * norm_dv * volume_element_comoving
        normalization *= 1/(kwargs_model['m_pivot']**plaw_index)

        draw_poisson = False
        self.mass_function = CDMPowerLaw(kwargs_model['log_mlow'], kwargs_model['log_mhigh'], plaw_index,
                                         draw_poisson, 10*normalization)
        self._plaw_index_theory = plaw_index


    def test_sampling(self):

        m = self.mass_function.draw()
        h, b = np.histogram(np.log10(m), range=(6, 8), bins=20)
        log10h = np.log10(h)
        log10m = b[0:-1]
        coefs = np.polyfit(log10m, log10h, 1)
        npt.assert_almost_equal((self._plaw_index_theory+1)/coefs[0], 1, 2)

if __name__ == '__main__':
   pytest.main()
