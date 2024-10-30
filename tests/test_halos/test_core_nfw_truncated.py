import numpy.testing as npt
import numpy as np
from lenstronomy.LensModel.Profiles.nfw_core_truncated import TNFWC as TNFWCLenstronomy
from lenstronomy.LensModel.Profiles.nfw import NFW as NFWLenstronomy
from pyHalo.Halos.HaloModels.NFW_core_trunc import TNFWCFieldHaloSIDM, TNFWCSubhaloSIDM, \
    rho_s_evolution, rs_evolution, rc_evolution
from pyHalo.Halos.HaloModels.NFW import NFWFieldHalo
from pyHalo.truncation_models import ConstantTruncationArcsec
from pyHalo.concentration_models import ConcentrationDiemerJoyce
from pyHalo.Halos.lens_cosmo import LensCosmo
import pytest


class TestTNFWC(object):

    def setup_method(self):

        mass = 10 ** 8
        self.lens_cosmo = LensCosmo(0.5, 2.5)
        self.tnfwc_lenstronomy = TNFWCLenstronomy()
        self.nfw_lenstronomy = NFWLenstronomy()
        self.truncation_class = ConstantTruncationArcsec(self.lens_cosmo, 1000.0)
        self.concentration_class = ConcentrationDiemerJoyce(self.lens_cosmo.cosmo, scatter=False)
        kwargs_profile = {'sidm_timescale': 1.0, 'lambda_t': 1.0}
        self.tnfwc = TNFWCFieldHaloSIDM(mass, 0.0, 0.0, None, 0.5, False,
                                        self.lens_cosmo, kwargs_profile, self.truncation_class,
                                        self.concentration_class, 1.0)

        self.tnfw_subhalo = TNFWCSubhaloSIDM(mass, 0.0, 0.0, None, 0.5, True,
                                        self.lens_cosmo, kwargs_profile, self.truncation_class,
                                        self.concentration_class, 1.0)

    def _get_profile(self, t, is_subhalo=False):

        mass = 10 ** 8
        kwargs_profile = {'sidm_timescale': 1.0, 'lambda_t': 1.0}
        if is_subhalo:
            profile = TNFWCSubhaloSIDM(mass, 0.0, 0.0, None, 0.5, is_subhalo,
                                         self.lens_cosmo, kwargs_profile, self.truncation_class,
                                         self.concentration_class, 1.0)
        else:
            profile = TNFWCFieldHaloSIDM(mass, 0.0, 0.0, None, 0.5, is_subhalo,
                                        self.lens_cosmo, kwargs_profile, self.truncation_class,
                                        self.concentration_class, 1.0)
        profile._halo_age = t
        return profile

    def _get_nfw_profile(self):

        mass = 10 ** 8
        self.lens_cosmo = LensCosmo(0.5, 2.5)
        kwargs_profile = {}
        profile = NFWFieldHalo(mass, 0.0, 0.0, None, 0.5, False,
                                        self.lens_cosmo, kwargs_profile, self.truncation_class,
                                        self.concentration_class, 1.0)
        return profile

    def test_mass_t0(self):

        profile = self._get_profile(0.0)
        nfw_profile = self._get_nfw_profile()
        mass3d = profile.mass_3d('r200')
        mass3d_nfw = nfw_profile.mass_3d('r200')
        npt.assert_almost_equal(mass3d_nfw/mass3d, 1.0, 4)
        npt.assert_almost_equal(mass3d/10**8, 1.0, 4.0)

    def test_rho_evolution(self):

        t_over_tc = 0.0
        rho = rho_s_evolution(t_over_tc)
        npt.assert_almost_equal(rho, 1.0, 5)

    def test_rs_evolution(self):

        t_over_tc = 0.0
        rs = rs_evolution(t_over_tc)
        npt.assert_almost_equal(rs, 1.0, 5)

    def test_rc_evolution(self):

        t_over_tc = 0.0
        rc = rc_evolution(t_over_tc)
        npt.assert_almost_equal(rc, 0.0, 5)

    def test_profile_evolution(self):

        profile0 = self._get_profile(0.0)
        rhos_0, rs_0, _ = profile0.nfw_params
        t_sidm = profile0.sidm_timescale

        t1 = 0.0 * t_sidm
        t2 = 0.5 * t_sidm
        t3 = 1.0 * t_sidm
        profile1 = self._get_profile(t1)
        profile2 = self._get_profile(t2)
        profile3 = self._get_profile(t3)

        kwargs_1 = profile1.lenstronomy_params[0][0]
        kwargs_2 = profile2.lenstronomy_params[0][0]
        kwargs_3 = profile3.lenstronomy_params[0][0]

        arcsec = 2 * np.pi / 360 / 3600
        dd_kpc = self.lens_cosmo.cosmo.D_A_z(profile0.z) * 1e3
        rs1 = rs_0 * rs_evolution(0.0)
        rs2 = rs_0 * rs_evolution(0.5)
        rs3 = rs_0 * rs_evolution(1.0)
        npt.assert_almost_equal(kwargs_1['Rs'] * dd_kpc * arcsec, rs1)
        npt.assert_almost_equal(kwargs_2['Rs'] * dd_kpc * arcsec, rs2)
        npt.assert_almost_equal(kwargs_3['Rs'] * dd_kpc * arcsec, rs3)

        rc1 = rs_0 * rc_evolution(0.0)
        rc2 = rs_0 * rc_evolution(0.5)
        rc3 = rs_0 * rc_evolution(1.0)
        npt.assert_almost_equal(kwargs_1['r_core'] * dd_kpc * arcsec, rc1, 5)
        npt.assert_almost_equal(kwargs_2['r_core'] * dd_kpc * arcsec, rc2, 5)
        npt.assert_almost_equal(kwargs_3['r_core'] * dd_kpc * arcsec, rc3, 5)

        rhos1 = rhos_0 * rho_s_evolution(0.0)
        rhos2 = rhos_0 * rho_s_evolution(0.5)
        rhos3 = rhos_0 * rho_s_evolution(1.0)
        (rho_s, rs_kpc, rc_kpc, rt_kpc, r200_0) = profile1.profile_args()
        npt.assert_almost_equal(rhos1, rho_s)
        (rho_s, rs_kpc, rc_kpc, rt_kpc, r200_0) = profile2.profile_args()
        npt.assert_almost_equal(rhos2, rho_s)
        (rho_s, rs_kpc, rc_kpc, rt_kpc, r200_0) = profile3.profile_args()
        npt.assert_almost_equal(rhos3, rho_s)

    def test_halo_age(self):

        is_subhalo = True
        kwargs_profile = {'sidm_timescale': 1.0, 'lambda_t': 1.0}
        profile1 = TNFWCSubhaloSIDM(10**8, 0.0, 0.0, None, 0.5, is_subhalo,
                                   self.lens_cosmo, kwargs_profile, self.truncation_class,
                                   self.concentration_class, 1.0)

        kwargs_profile = {'sidm_timescale': 1.0, 'lambda_t': 2.0}
        profile2 = TNFWCSubhaloSIDM(10 ** 8, 0.0, 0.0, None, 0.5, is_subhalo,
                                    self.lens_cosmo, kwargs_profile, self.truncation_class,
                                    self.concentration_class, 1.0)
        npt.assert_equal(profile1.halo_effective_age < profile2.halo_effective_age, True)

        is_subhalo = False
        kwargs_profile = {'sidm_timescale': 1.0, 'lambda_t': 1.0}
        profile1 = TNFWCFieldHaloSIDM(10 ** 8, 0.0, 0.0, None, 0.5, is_subhalo,
                                    self.lens_cosmo, kwargs_profile, self.truncation_class,
                                    self.concentration_class, 1.0)

        kwargs_profile = {'sidm_timescale': 1.0, 'lambda_t': 2.0}
        profile2 = TNFWCFieldHaloSIDM(10 ** 8, 0.0, 0.0, None, 0.5, is_subhalo,
                                    self.lens_cosmo, kwargs_profile, self.truncation_class,
                                    self.concentration_class, 1.0)
        npt.assert_equal(profile1.halo_effective_age, profile2.halo_effective_age)

    def test_mass_evolution(self):

        m = []
        halo_ages = np.linspace(0.0, 1.6, 100)

        for i, ti in enumerate(halo_ages):
            prof = self._get_profile(ti)
            if i==0:
                r200_0 = prof.nfw_params[2]
            m.append(prof.mass_3d(r200_0))
        m = np.array(m)/m[0]
        npt.assert_allclose(m, 1.0, 1)

if __name__ == '__main__':
    pytest.main()
