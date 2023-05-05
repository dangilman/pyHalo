import numpy.testing as npt
from pyHalo.Cosmology.geometry import Geometry
from pyHalo.Cosmology.cosmology import Cosmology
import astropy.units as un
import numpy as np
import pytest

class TestCosmology(object):

    def setup_method(self):

        self.arcsec = 2 * np.pi / 360 / 3600
        self.zlens = 1
        self.zsource = 2
        self.angle_diameter = 2/self.arcsec
        self.angle_radius = 0.5*self.angle_diameter

        self.H0 = 70
        self.omega_baryon = 0.03
        self.omega_DM = 0.25
        self.sigma8 = 0.82
        curvature = 'flat'
        self.ns = 0.9608
        cosmo_params = {'H0': self.H0, 'Om0': self.omega_baryon + self.omega_DM, 'Ob0': self.omega_baryon,
                      'sigma8': self.sigma8, 'ns': self.ns, 'curvature': curvature}
        self._dm, self._bar = self.omega_DM, self.omega_baryon
        self.cosmo = Cosmology(cosmo_kwargs=cosmo_params)
        self.geometry = Geometry(self.cosmo, self.zlens, self.zsource, self.angle_diameter, 'DOUBLE_CONE')

    def test_cosmo(self):

        da_true = self.cosmo.D_A(0, 1.824)
        da_interp = self.cosmo.D_A_z(1.824)
        npt.assert_almost_equal(da_true/da_interp, 1, 5)

        dc_true = self.cosmo.astropy.comoving_transverse_distance(1.4).value
        dc_interp = self.cosmo.D_C_z(1.4)
        dc_astropy = self.cosmo.astropy.comoving_distance(1.4).value
        npt.assert_almost_equal(dc_true/dc_interp, 1)
        npt.assert_almost_equal(dc_astropy/dc_true, 1)

        dc_transverse = self.cosmo.D_C_transverse(0.8)
        dc = self.cosmo.astropy.comoving_transverse_distance(0.8).value
        npt.assert_almost_equal(dc/dc_transverse, 1)

        ez = self.cosmo.E_z(0.8)
        ez_astropy = self.cosmo.astropy.efunc(0.8)
        npt.assert_almost_equal(ez / ez_astropy, 1)

        kpc_per_asec = self.cosmo.kpc_proper_per_asec(0.5)
        kpc_per_arcsec_true = self.cosmo.astropy.kpc_proper_per_arcmin(0.5).value / 60
        npt.assert_almost_equal(kpc_per_asec, kpc_per_arcsec_true, 2)

        rho_crit_0 = self.cosmo.rho_crit(0)
        rho_pc = un.Quantity(self.cosmo.astropy.critical_density(0), unit=un.Msun / un.pc ** 3)
        rho_Mpc = rho_pc.value * (1e+6) ** 3
        npt.assert_almost_equal(rho_crit_0/rho_Mpc, 1, 3)

        rho_crit = self.cosmo.rho_crit(0.6)
        rho_pc = un.Quantity(self.cosmo.astropy.critical_density(0.6), unit=un.Msun / un.m ** 3)
        rho_Mpc = rho_pc.value * self.cosmo.Mpc ** 3
        npt.assert_almost_equal(rho_crit / rho_Mpc, 1, 3)

        rho_crit_dark_matter = self.cosmo.rho_dark_matter_crit
        rho_crit_DM_astropy = self.cosmo.astropy.critical_density(0.).value * \
                              self.cosmo.density_to_MsunperMpc * self.cosmo.astropy.Odm(0.)
        npt.assert_almost_equal(rho_crit_DM_astropy, rho_crit_dark_matter)

        rho_crit = self.cosmo.rho_crit(0.3)
        rho_pc = un.Quantity(self.cosmo.astropy.critical_density(0.3), unit=un.Msun / un.pc ** 3)
        rho_Mpc = rho_pc.value * (1e+6) ** 3
        npt.assert_almost_equal(rho_crit / rho_Mpc, 1, 3)

        colossus = self.cosmo.colossus

        npt.assert_almost_equal(colossus.Om0, self.omega_DM + self.omega_baryon)
        npt.assert_almost_equal(colossus.Ob0, self.omega_baryon)
        npt.assert_almost_equal(colossus.H0, self.H0)
        npt.assert_almost_equal(colossus.sigma8, self.sigma8)
        npt.assert_almost_equal(colossus.ns, self.ns)

        halo_collapse_z = 9.
        age_today = self.cosmo.astropy.age(0.).value
        age_z = self.cosmo.astropy.age(halo_collapse_z).value
        age = age_today - age_z
        npt.assert_almost_equal(age, self.cosmo.halo_age(0., zform=halo_collapse_z))

        npt.assert_almost_equal(self.cosmo.scale_factor(0.7), self.cosmo.astropy.scale_factor(0.7))

if __name__ == '__main__':
     pytest.main()
