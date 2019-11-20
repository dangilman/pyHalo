import numpy.testing as npt
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Halos.lens_cosmo import LensCosmo
import numpy as np
import pytest

class TestLensCosmo(object):

    def setup(self):

        H0 = 70
        omega_baryon = 0.03
        omega_DM = 0.25
        sigma8 = 0.82
        curvature = 'flat'
        ns = 0.9608
        cosmo_params = {'H0': H0, 'Om0': omega_baryon + omega_DM, 'Ob0': omega_baryon,
                      'sigma8': sigma8, 'ns': ns, 'curvature': curvature}
        self._dm, self._bar = omega_DM, omega_baryon
        self.cosmo = Cosmology(cosmo_kwargs=cosmo_params)

        zlens, zsource = 0.5, 1.5
        self.lens_cosmo = LensCosmo(zlens, zsource, self.cosmo)

    def test_routines(self):

        m_thermal = 3.3
        mhm = self.lens_cosmo.mthermal_to_halfmode(m_thermal)
        m_thermal_2 = self.lens_cosmo.halfmode_to_thermal(mhm)

        npt.assert_almost_equal(mhm/10**8.478, 1, 3)
        npt.assert_almost_equal(m_thermal/m_thermal_2, 1, 3)

        eps_crit = self.lens_cosmo.get_epsiloncrit(0.5, 1.5)
        npt.assert_almost_equal(eps_crit/10**15.3611, 1, 4)
        eps_crit_asec = self.lens_cosmo.get_sigmacrit_z1z2(0.5, 1.5)
        eps_crit_kpc = eps_crit_asec * self.cosmo.kpc_per_asec(0.5) ** -2
        npt.assert_almost_equal(eps_crit_kpc / 10 ** 15.3611, 0.001**2, 4)


