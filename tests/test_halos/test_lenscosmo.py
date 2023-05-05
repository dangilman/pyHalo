import numpy.testing as npt
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Halos.lens_cosmo import LensCosmo
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM
from astropy.constants import G, c
from pyHalo.Halos.concentration import ConcentrationDiemerJoyce
import astropy.units as un
from pyHalo.Halos.HaloModels.NFW import NFWSubhhalo, NFWFieldHalo
from pyHalo.Cosmology.cosmology import Cosmology
from lenstronomy.Cosmo.lens_cosmo import LensCosmo as LensCosmoLenstronomy
from pyHalo.Halos.HaloModels.generalized_nfw import GeneralNFWSubhalo

class TestLensCosmo(object):

    def setup_method(self):

        kwargs_cosmo = {'Om0': 0.2}
        self.cosmo = Cosmology(cosmo_kwargs=kwargs_cosmo)
        zlens, zsource = 0.3, 1.7
        self.lens_cosmo = LensCosmo(zlens, zsource, self.cosmo)
        self.h = self.cosmo.h

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

    def test_subhalo_accretion(self):

        zi = [self.lens_cosmo.z_accreted_from_zlens(10**8, 0.5)
              for _ in range(0, 20000)]

        h, b = np.histogram(zi, bins=np.linspace(0.5, 6, 20))

        # number at the lens redshift should be about 5x that at redshift 4
        ratio = h[0]/h[12]
        npt.assert_almost_equal(ratio/5 - 1, 0., 1)

    def test_mhm_convert(self):

        mthermal = 5.3
        mhm = self.lens_cosmo.mthermal_to_halfmode(mthermal)
        mthermal_out = self.lens_cosmo.halfmode_to_thermal(mhm)
        npt.assert_almost_equal(mthermal/mthermal_out, 1, 2)

        fsl = self.lens_cosmo.mhm_to_fsl(10**8.)
        npt.assert_array_less(fsl, 100)

    def test_nfw_definitions_wrt_lenstronomy(self):

        astropy = FlatLambdaCDM(70, 0.3)
        cosmo = Cosmology(astropy)
        zlens = 0.5
        zsource = 2.0
        lc = LensCosmoLenstronomy(zlens, zsource, astropy)
        lens_cosmo = LensCosmo(zlens, zsource, cosmo)

        rho0_lenstronomy, rs_lenstronomy, r200_lenstronomy = lc.nfwParam_physical(10 ** 8, 16.0)
        rho0, rs, r200 = lens_cosmo.nfwParam_physical(10 ** 8, 16.0, zlens)
        npt.assert_almost_equal(rho0_lenstronomy/rho0, 1, 4)
        npt.assert_almost_equal(rs/rs_lenstronomy, 1.0, 4)
        npt.assert_almost_equal(r200/r200_lenstronomy, 1.0, 4)

        rho0_kpc, rs_kpc, r200_kpc = lens_cosmo.NFW_params_physical(10 ** 8, 16.0, zlens)
        npt.assert_almost_equal(rho0_kpc, rho0*1000**-3)
        npt.assert_almost_equal(rs_kpc, rs*1000)
        npt.assert_almost_equal(r200_kpc, r200 * 1000)


if __name__ == '__main__':
    pytest.main()


