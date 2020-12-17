import numpy.testing as npt
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Halos.lens_cosmo import LensCosmo
import numpy as np
import pytest
from scipy.integrate import quad
from astropy.constants import G, c
from pyHalo.Halos.concentration import Concentration
import astropy.units as un

class TestLensCosmo(object):

    def setup(self):

        kwargs_cosmo = {'Om0': 0.2}
        self.cosmo = Cosmology(cosmo_kwargs=kwargs_cosmo)
        zlens, zsource = 0.3, 1.7
        self.lens_cosmo = LensCosmo(zlens, zsource, self.cosmo)
        self.h = self.cosmo.h
        self.con = Concentration(self.lens_cosmo)

    def test_const(self):

        D_ds = self.cosmo.D_A(0.3, 1.7)
        D_d = self.cosmo.D_A_z(0.3)
        D_s = self.cosmo.D_A_z(1.7)

        c_Mpc_sec = un.Quantity(c, unit=un.Mpc / un.s)
        G_Mpc3_Msun_sec2 = un.Quantity(G, unit=un.Mpc ** 3 / un.s ** 2 / un.solMass)

        const = c_Mpc_sec ** 2 / (4 * np.pi * G_Mpc3_Msun_sec2)
        sigma_crit_mpc = const.value * D_s / (D_d * D_ds)
        sigma_crit_kpc = sigma_crit_mpc * 1000 ** -2

        npt.assert_almost_equal(self.lens_cosmo.sigma_crit_lensing/sigma_crit_mpc, 1, 4)
        npt.assert_almost_equal(self.lens_cosmo.sigma_crit_lens_kpc/sigma_crit_kpc, 1, 4)

    def test_sigma_crit_mass(self):

        area = 2.
        sigma_crit_mass = self.lens_cosmo.sigma_crit_mass(0.7, area)
        sigma_crit = self.lens_cosmo.get_sigma_crit_lensing(0.7, 1.7)
        npt.assert_almost_equal(sigma_crit_mass, sigma_crit * area)

    def test_colossus(self):

        colossus = self.lens_cosmo.colossus
        npt.assert_almost_equal(colossus.Om0, 0.2)

    def test_truncate_roche(self):

        m = 10**9.
        norm = 1.4
        power = 0.9
        r3d = 1000

        rt = norm * (m / 10 ** 7) ** (1./3) * (r3d / 50) ** power
        rtrunc = self.lens_cosmo.truncation_roche(m, r3d, norm, power)
        npt.assert_almost_equal(rt, rtrunc, 3)

    def test_LOS_trunc(self):

        rt = self.lens_cosmo.LOS_truncation_rN(10**8, 0.2, 90)
        rtrunc = self.lens_cosmo.rN_M_nfw_comoving(10**8 * self.lens_cosmo.cosmo.h, 90, 0.2)
        npt.assert_almost_equal(rt, 1000 * rtrunc * (1+0.2)**-1 / self.lens_cosmo.cosmo.h)

    def test_NFW_concentration(self):

        c = self.lens_cosmo.NFW_concentration(10 ** 9, 0.2, model='diemer19', scatter=False)
        c2 = self.lens_cosmo.NFW_concentration(10 ** 9, 0.2, model='diemer19', scatter=True)
        npt.assert_raises(AssertionError, npt.assert_array_equal, c, c2)

        logmhm = 8.
        c_wdm = self.lens_cosmo.NFW_concentration(10 ** 9, 0.2, model='diemer19', scatter=False, logmhm=logmhm,
                                                  c_scale=100, c_power=-0.4)

        suppresion = self.con.WDM_concentration_suppresion_factor(10**9, 0.2, logmhm, 100, -0.4)
        npt.assert_almost_equal(suppresion * c, c_wdm)

    def test_subhalo_accretion(self):

        zi = [self.lens_cosmo.z_accreted_from_zlens(10**8, 0.5)
              for _ in range(0, 6000)]

        h, b = np.histogram(zi, bins=np.linspace(0.5, 6, 20))

        # number at the lens redshift should be about 5x that at redshift 4
        ratio = h[0]/h[12]
        npt.assert_almost_equal(ratio/5 - 1, 0., 1)

    def test_nfw_fundamental_parameters(self):

        for z in [0., 0.74]:

            M, c = 10**8, 17.5
            rho_crit_z = self.cosmo.rho_crit(z)

            rhos, rs, r200 = self.lens_cosmo._nfwParam_physical_Mpc(M, c, z)

            def _profile(x):
                fac = x * (1 + x) ** 2
                return 1. / fac
            def _integrand(x):
                return 4 * np.pi * x ** 2 * _profile(x)

            volume = 4 * np.pi/3 * r200 ** 3
            integral = quad(_integrand, 0, r200/rs)[0]
            mean_density = rhos * rs ** 3 * integral / volume
            ratio = mean_density/rho_crit_z

            npt.assert_almost_equal(ratio/200, 1., 3)

if __name__ == '__main__':
    pytest.main()


