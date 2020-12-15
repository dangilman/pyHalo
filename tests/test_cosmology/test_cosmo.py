import numpy.testing as npt
from pyHalo.Cosmology.geometry import Geometry
from pyHalo.Cosmology.cosmology import Cosmology
import astropy.units as un
import numpy as np
import pytest

class TestCosmology(object):

    def setup(self):

        self.arcsec = 2 * np.pi / 360 / 3600
        self.zlens = 1
        self.zsource = 2
        self.angle_diameter = 2/self.arcsec
        self.angle_radius = 0.5*self.angle_diameter

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
        self.geometry = Geometry(self.cosmo, self.zlens, self.zsource, self.angle_diameter, 'DOUBLE_CONE')

    def test_cosmo(self):

        da_true = self.cosmo.D_A(0, 1.824)
        da_interp = self.cosmo.D_A_z(1.824)
        npt.assert_almost_equal(da_true/da_interp, 1, 5)

        dc_true = self.cosmo.D_C(1.4)
        dc_interp = self.cosmo.D_C_z((1.4))
        dc_astropy = self.cosmo.astropy.comoving_distance(1.4).value
        npt.assert_almost_equal(dc_true/dc_interp, 1)
        npt.assert_almost_equal(dc_astropy/dc_true, 1)

        dc_transverse = self.cosmo.D_C_transverse(0.8)
        dc = self.cosmo.D_C(0.8)
        npt.assert_almost_equal(dc/dc_transverse, 1)

        dc_12_true = self.cosmo.D_C_transverse(1.) - self.cosmo.D_C_transverse(0.5)
        dc_12_interp = self.cosmo.D_C_z12(0.5, 1)
        npt.assert_almost_equal(dc_12_interp, dc_12_true)

        ez = self.cosmo.E_z(0.8)
        ez_astropy = self.cosmo.astropy.efunc(0.8)
        npt.assert_almost_equal(ez / ez_astropy, 1)

        kpc_per_asec = self.cosmo.kpc_per_asec(0.5)
        npt.assert_almost_equal(kpc_per_asec, 6.147, 3)

        rho_crit_0 = self.cosmo.rho_crit(0)
        rho_pc = un.Quantity(self.cosmo.astropy.critical_density(0), unit=un.Msun / un.pc ** 3)
        rho_Mpc = rho_pc.value * (1e+6) ** 3
        npt.assert_almost_equal(rho_crit_0/rho_Mpc, 1, 3)

        rho_crit = self.cosmo.rho_crit(0.6)
        rho_pc = un.Quantity(self.cosmo.astropy.critical_density(0.6), unit=un.Msun / un.pc ** 3)
        rho_Mpc = rho_pc.value * (1e+6) ** 3
        npt.assert_almost_equal(rho_crit / rho_Mpc, 1, 3)


if __name__ == '__main__':
     pytest.main()
