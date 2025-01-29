import numpy.testing as npt
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Halos.accretion import InfallDistributionGalacticus2024, InfallDistributionHybrid
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM
from astropy.constants import G, c
import astropy.units as un
from pyHalo.Cosmology.cosmology import Cosmology
from lenstronomy.Cosmo.lens_cosmo import LensCosmo as LensCosmoLenstronomy


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

    def test_infall(self):

        zlens = 0.5
        zsource = 2.0

        lens_cosmo = LensCosmo(zlens, zsource, self.cosmo, infall_redshift_model='HYBRID_INFALL')
        z_infall = lens_cosmo.z_accreted_from_zlens(10 ** 8)
        npt.assert_equal(True, z_infall > zlens)

        lens_cosmo = LensCosmo(zlens, zsource, self.cosmo, infall_redshift_model='DIRECT_INFALL')
        z_infall = lens_cosmo.z_accreted_from_zlens(10 ** 8)
        npt.assert_equal(True, z_infall > zlens)

        infall_time_model = InfallDistributionGalacticus2024
        kwargs_infall_model = {}
        lens_cosmo = LensCosmo(zlens, zsource, self.cosmo, infall_redshift_model=infall_time_model,
                               kwargs_infall_model=kwargs_infall_model)
        z_infall = lens_cosmo.z_accreted_from_zlens(10 ** 8)
        npt.assert_equal(True, z_infall > zlens)

        infall_time_model = InfallDistributionHybrid
        kwargs_infall_model = {'log_m_host': 13.0}
        lens_cosmo = LensCosmo(zlens, zsource, self.cosmo, infall_redshift_model=infall_time_model,
                               kwargs_infall_model=kwargs_infall_model)
        z_infall = lens_cosmo.z_accreted_from_zlens(10 ** 8)
        npt.assert_equal(True, z_infall > zlens)

    def test_sidm_halo_age(self):

        z, z_infall, lambda_t, zform = 0.5, 5.0, 1.0, 10.0
        halo_age = self.lens_cosmo.sidm_halo_effective_age(z, z_infall, lambda_t, zform=10.0)
        halo_age_2 = self.lens_cosmo.sidm_halo_effective_age(z, z_infall, 2*lambda_t, zform=10.0)
        npt.assert_equal(halo_age_2 > halo_age, True)

    def test_nfw_params(self):

        m, c, z = 10**8, 11.0, 0.5
        rhos, rs, r200 = self.lens_cosmo.nfwParam_physical(m,c,z)
        Rs_Angle, Theta_Rs = self.lens_cosmo.nfw_physical2angle(m,c,z)
        rs_angle, theta_rs = self.lens_cosmo.nfw_physical2angle_fromNFWparams(rhos, rs, z)
        npt.assert_almost_equal(Rs_Angle, rs_angle)
        npt.assert_almost_equal(Theta_Rs, theta_rs)

        rhos_kpc, rs_kpc, r200_kpc = self.lens_cosmo.NFW_params_physical(m, c, z)
        npt.assert_almost_equal(rhos_kpc, rhos * 1e-9)
        npt.assert_almost_equal(rs_kpc, rs * 1e3)
        rs_angle, theta_rs = self.lens_cosmo.nfw_physical2angle_fromNFWparams(rhos_kpc * 1e9,
                                                                              rs_kpc * 1e-3,
                                                                              z)
        npt.assert_almost_equal(Rs_Angle, rs_angle)
        npt.assert_almost_equal(Theta_Rs, theta_rs)

    def test_sidm_timescale(self):

        # test scaling with halo mass
        m1 = 10 ** 10
        c1 = 5
        rhos, rs, r200 = self.lens_cosmo.NFW_params_physical(m1, c1, 0.5)
        tc1 = self.lens_cosmo.sidm_collapse_timescale(rhos, rs, 10.0)
        m2 = 10 ** 8
        c2 = 5
        rhos, rs, r200 = self.lens_cosmo.NFW_params_physical(m2, c2, 0.5)
        tc2 = self.lens_cosmo.sidm_collapse_timescale(rhos, rs, 10.0)
        npt.assert_almost_equal((m2/m1) ** (-1/3), tc2/tc1, 6)

        # test scaling with concentration
        m1 = 10 ** 8
        c1 = 10
        rhos, rs, r200 = self.lens_cosmo.NFW_params_physical(m1, c1, 0.5)
        tc1 = self.lens_cosmo.sidm_collapse_timescale(rhos, rs, 10.0)
        m2 = 10 ** 8
        c2 = 20
        rhos, rs, r200 = self.lens_cosmo.NFW_params_physical(m2, c2, 0.5)
        tc2 = self.lens_cosmo.sidm_collapse_timescale(rhos, rs, 10.0)
        npt.assert_almost_equal((c2 / c1) ** (-7 / 2), tc2 / tc1, 1)

        rhos = 2.74e8
        rs = 0.141
        tc_yangetal_2023 = self.lens_cosmo.sidm_collapse_timescale(rhos, rs, 7.1)
        npt.assert_almost_equal(tc_yangetal_2023, 28.07, 2)


if __name__ == '__main__':
    pytest.main()

