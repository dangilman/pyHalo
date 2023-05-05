import pytest
import numpy as np
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Cosmology.geometry import Geometry
import numpy.testing as npt
from pyHalo.Rendering.MassFunctions.delta_function import DeltaFunction

class TestDeltaFunction(object):

    def setup_method(self):

        z = 0.5
        delta_z = 0.02
        opening_angle = 6.0
        zlens = 0.5
        zsource = 1.5
        cosmo = Cosmology()
        astropy = cosmo.astropy
        geometry_class = Geometry(cosmo, zlens, zsource, opening_angle, 'DOUBLE_CONE')
        density_to_MsunperMpc = geometry_class.cosmo.density_to_MsunperMpc
        self._rho = astropy.Odm(0.) * astropy.critical_density(z).value * density_to_MsunperMpc
        self._volume = geometry_class.volume_element_comoving(z, delta_z)
        self._mass = 10 ** 7
        mass_fraction = 1.0
        self._kwargs_model = {'mass_fraction': mass_fraction, 'mass': self._mass,
                              'draw_poisson': False, 'LOS_normalization': 1.0}
        draw_poisson = False
        self.mass_function = DeltaFunction(self._mass, self._volume, mass_fraction*self._rho, draw_poisson)
        draw_poisson = True
        self.mass_function_poisson = DeltaFunction(self._mass, self._volume, mass_fraction*self._rho, draw_poisson)
        self._kwargs_from_redshift = {'z': z, 'delta_z': delta_z, 'geometry_class': geometry_class,
                                      'kwargs_model': self._kwargs_model}

    def test_mass_function(self):

        m = self.mass_function.draw()
        npt.assert_equal(m[0]==self._mass, True)
        n_theory = np.round(self._kwargs_model['mass_fraction']*self._rho * self._volume/self._mass)
        npt.assert_equal(True, len(m)==n_theory)

    def test_draw_poisson(self):

        m = self.mass_function.draw()
        m_poisson = self.mass_function_poisson.draw()
        npt.assert_equal(True, len(m)!=len(m_poisson))

    def test_from_redshift(self):

        mfunc = DeltaFunction.from_redshift(**self._kwargs_from_redshift)
        m = mfunc.draw()
        npt.assert_equal(True, len(m)==len(self.mass_function.draw()))

if __name__ == '__main__':
   pytest.main()
