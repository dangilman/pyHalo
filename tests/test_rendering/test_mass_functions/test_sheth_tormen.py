import pytest
import numpy as np
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Cosmology.geometry import Geometry
import numpy.testing as npt
from pyHalo.Rendering.MassFunctions.density_peaks import ShethTormen, ShethTormenTurnover, ShethTormenMixedWDM
from pyHalo.Rendering.MassFunctions.mass_function_base import CDMPowerLaw, WDMPowerLaw, MixedWDMPowerLaw
from pyHalo.Rendering.MassFunctions.density_peaks import evaluate_mass_function
from colossus.lss.mass_function import massFunction
from colossus.cosmology import cosmology

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
        kwargs_model['m_pivot'] = 10.0 ** 8
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

        npt.assert_equal(False, self.mass_function._draw_poisson)
        npt.assert_equal(True, self.mass_function_poisson._draw_poisson)

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

    def test_galacticus_comparison(self):

        # first z = 0
        galacticus_dndm = np.array([5.05460310e-06, 3.25319529e-06, 2.09420639e-06, 1.34837633e-06,
       8.68332355e-07, 5.59300208e-07, 3.60324589e-07, 2.32176844e-07,
       1.49635804e-07, 9.64583113e-08, 6.21916467e-08, 4.01068273e-08,
       2.58695766e-08, 1.66897785e-08, 1.07696749e-08, 6.95097691e-09,
       4.48725449e-09, 2.89739324e-09, 1.87122600e-09, 1.20875189e-09,
       7.80980403e-10])

        h = 0.675
        cosmo_kwargs = {'H0': h * 100,
                        'Om0': 0.26 + 0.049,
                        'Ob0': 0.049,
                        'ns': 0.965,
                        'sigma8': 0.81,
                        'power_law': False}
        _ = cosmology.setCosmology('custom', cosmo_kwargs)
        m = np.logspace(7,9,len(galacticus_dndm))
        z = 0.0
        # colossus version 1.3.8 and 1.3.6 give same answer
        pyHalo_dndm = evaluate_mass_function(m, h, z, 'sheth99')
        npt.assert_almost_equal(pyHalo_dndm / galacticus_dndm, 1, 2)

        # z = 0.5
        galacticus_dndm = np.array([5.80168983e-06, 3.73347001e-06, 2.40295113e-06, 1.54686275e-06,
       9.95941001e-07, 6.41343627e-07, 4.13069941e-07, 2.66092273e-07,
       1.71441890e-07, 1.10478483e-07, 7.12056564e-08, 4.59015330e-08,
       2.95948083e-08, 1.90844274e-08, 1.23088498e-08, 7.94017102e-09,
       5.12289474e-09, 3.30577390e-09, 2.13354466e-09, 1.37720854e-09,
       8.89130082e-10])

        h = 0.675
        cosmo_kwargs = {'H0': h * 100,
                        'Om0': 0.26 + 0.049,
                        'Ob0': 0.049,
                        'ns': 0.965,
                        'sigma8': 0.81,
                        'power_law': False}
        _ = cosmology.setCosmology('custom', cosmo_kwargs)
        m = np.logspace(7, 9, len(galacticus_dndm))
        z = 0.5
        # colossus version 1.3.8 and 1.3.6 give same answer
        pyHalo_dndm = evaluate_mass_function(m, h, z, 'sheth99')
        npt.assert_almost_equal(pyHalo_dndm / galacticus_dndm, 1, 2)

        # z = 2.0
        galacticus_dndm = np.array([7.74354495e-06, 4.97311423e-06, 3.19404677e-06, 2.05152021e-06,
       1.31773849e-06, 8.46444536e-07, 5.43726191e-07, 3.49277636e-07,
       2.24370641e-07, 1.44132382e-07, 9.25875345e-08, 5.94748797e-08,
       3.82031256e-08, 2.45381672e-08, 1.57600025e-08, 1.01212409e-08,
       6.49929481e-09, 4.17296374e-09, 2.67891506e-09, 1.71948492e-09,
       1.10344583e-09])

        h = 0.675
        cosmo_kwargs = {'H0': h * 100,
                        'Om0': 0.26 + 0.049,
                        'Ob0': 0.049,
                        'ns': 0.965,
                        'sigma8': 0.81,
                        'power_law': False}
        _ = cosmology.setCosmology('custom', cosmo_kwargs)
        m = np.logspace(7, 9, len(galacticus_dndm))
        z = 2.0
        # colossus version 1.3.8 and 1.3.6 give same answer
        pyHalo_dndm = evaluate_mass_function(m, h, z, 'sheth99')
        npt.assert_almost_equal(pyHalo_dndm / galacticus_dndm, 1, 2)

if __name__ == '__main__':
   pytest.main()
