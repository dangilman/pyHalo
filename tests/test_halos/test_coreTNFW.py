import numpy.testing as npt
import numpy as np
from pyHalo.Halos.HaloModels.coreTNFW import coreTNFWFieldHalo, coreTNFWSubhalo
from pyHalo.Halos.HaloModels.TNFW import TNFWSubhalo
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Cosmology.cosmology import Cosmology
from colossus.halo.concentration import concentration
import pytest
from lenstronomy.LensModel.Profiles.tnfw import TNFW

class TestcoreTNFWHalos(object):

    def setup(self):

        mass = 10**8.
        x = 0.5
        y = 1.
        r3d = np.sqrt(1 + 0.5 ** 2 + 70**2)
        self.r3d = r3d
        mdef = 'coreTNFW'
        self.z = 0.25
        sub_flag = True

        self.H0 = 70
        self.omega_baryon = 0.03
        self.omega_DM = 0.25
        self.sigma8 = 0.82
        curvature = 'flat'
        self.ns = 0.9608
        cosmo_params = {'H0': self.H0, 'Om0': self.omega_baryon + self.omega_DM, 'Ob0': self.omega_baryon,
                        'sigma8': self.sigma8, 'ns': self.ns, 'curvature': curvature}
        self._dm, self._bar = self.omega_DM, self.omega_baryon
        cosmo = Cosmology(cosmo_kwargs=cosmo_params)
        self.lens_cosmo = LensCosmo(self.z, 2., cosmo)

        cross_section_type = 'POWER_LAW'
        self._cross_norm = 5
        cross_section_kwargs = {'norm': self._cross_norm, 'v_dep': 0.5, 'v_ref': 30.}
        profile_args = {'RocheNorm': 1.2, 'RocheNu': 2 / 3,
                        'evaluate_mc_at_zlens': True,
                        'log_mc': None, 'c_scale': 60.,
                        'c_power': -0.17, 'c_scatter': False,
                        'mc_model': 'diemer19', 'LOS_truncation_factor': 40,
                        'c_scatter_dex': 0.1, 'mc_mdef': '200c',
                        'cross_section_type': cross_section_type,
                        'kwargs_cross_section': cross_section_kwargs,
                        'numerical_deflection_angle_class': self._deflection_function,
                        'SIDM_rhocentral_function': self._rho_function}


        self.profile_args = profile_args

        self.mass = mass
        self.subhalo = coreTNFWSubhalo(mass, x, y, r3d, mdef, self.z,
                                   sub_flag, self.lens_cosmo,
                                   profile_args, unique_tag=np.random.rand())


        self.field_halo = coreTNFWFieldHalo(mass, x, y, r3d, mdef, self.z,
                            sub_flag, self.lens_cosmo,
                            profile_args, unique_tag=np.random.rand())

    def _deflection_function(self, x, y, rs, r_core, r_trunc, norm):

        tnfw = TNFW()

        return tnfw.derivatives(x, y, rs, norm, r_trunc)

    def _rho_function(self, m, z, delta_c_dex, cross_section_type, kwargs_cross_section):

        return kwargs_cross_section['norm']

    def test_lenstronomy_kwargs(self):

        prof = self.field_halo
        (c, rtrunc_kpc, rho_central) = prof.profile_args
        kwargs, func = prof.lenstronomy_params
        rs_angle, theta_rs = self.lens_cosmo.nfw_physical2angle(prof.mass, c, prof.z)
        rtrunc_angle = rtrunc_kpc / self.lens_cosmo.cosmo.kpc_proper_per_asec(prof.z)
        rhos_kpc, rs_kpc, _ = self.subhalo.lens_cosmo.NFW_params_physical(prof.mass, c, prof._tnfw.z_eval)

        npt.assert_almost_equal(prof.x, kwargs[0]['center_x'])
        npt.assert_almost_equal(prof.y, kwargs[0]['center_y'])
        npt.assert_almost_equal(rtrunc_angle, kwargs[0]['r_trunc'])
        npt.assert_almost_equal(rs_angle, kwargs[0]['Rs'])

        x, y, rs, r_core, r_trunc, norm = 0.1, 0.1, 1., 1., 0.5, 1.
        out = func(x, y, rs, r_core, r_trunc, norm)
        npt.assert_almost_equal(out, self._deflection_function(x, y, rs, r_core, r_trunc, norm))

    def test_lenstronomy_ID(self):

        id = self.subhalo.lenstronomy_ID
        npt.assert_string_equal(id[0], 'NumericalAlpha')
        id = self.field_halo.lenstronomy_ID
        npt.assert_string_equal(id[0], 'NumericalAlpha')

    def test_z_infall(self):

        z_eval = self.subhalo._tnfw.z_eval
        npt.assert_equal(True, self.z <= z_eval)

        z_eval = self.field_halo._tnfw.z_eval
        npt.assert_equal(True, self.z == z_eval)

    def test_profile_args(self):

        profile_args = self.subhalo.profile_args
        (c, rt, rho0) = profile_args
        con = concentration(self.lens_cosmo.cosmo.h * self.mass, '200c', self.z,
                            model='diemer19')
        npt.assert_almost_equal(c/con, 1, 2)
        trunc = self.profile_args['RocheNorm'] * (10 ** 8 / 10 ** 7) ** (1. / 3) * \
                (self.r3d / 50) ** self.profile_args['RocheNu']
        npt.assert_almost_equal(trunc, rt, 3)
        rho_central = self._cross_norm
        npt.assert_almost_equal(rho0, rho_central)

        profile_args = self.field_halo.profile_args
        (c, rt, rho0) = profile_args
        con = concentration(self.lens_cosmo.cosmo.h * self.mass, '200c', self.z,
                            model='diemer19')
        npt.assert_almost_equal(c / con, 1, 2)

        m_h = self.mass * self.lens_cosmo.cosmo.h
        r50_comoving = self.lens_cosmo.rN_M_nfw_comoving(m_h, self.profile_args['LOS_truncation_factor'], self.z)
        r50_physical = r50_comoving * self.lens_cosmo.cosmo.scale_factor(self.z) / self.lens_cosmo.cosmo.h
        r50_physical_kpc = r50_physical * 1000
        npt.assert_almost_equal(r50_physical_kpc, rt)

        rho_central = self._cross_norm
        npt.assert_almost_equal(rho0, rho_central)

    def test_fromTNFW(self):

        tnfw = TNFWSubhalo(self.subhalo.mass, self.subhalo.x, self.subhalo.y, self.subhalo.r3d, 'TNFW', self.subhalo.z, True,
                           self.subhalo.lens_cosmo, self.profile_args, unique_tag=1.23)

        coreTNFW = coreTNFWSubhalo.fromTNFW(tnfw, kwargs_new=self.profile_args)

        npt.assert_almost_equal(coreTNFW.mass, tnfw.mass)
        npt.assert_almost_equal(coreTNFW.x, tnfw.x)
        npt.assert_almost_equal(coreTNFW.y, tnfw.y)
        npt.assert_almost_equal(coreTNFW.r3d, tnfw.r3d)

        prof_params = tnfw.profile_args
        prof_params_core = coreTNFW.profile_args
        for i in range(0, len(prof_params)-1):
            npt.assert_almost_equal(prof_params[i], prof_params_core[i])

        lenstronomy_params_tnfw, _ = tnfw.lenstronomy_params
        lenstronomy_params_coretnfw, _ = coreTNFW.lenstronomy_params

        npt.assert_almost_equal(lenstronomy_params_tnfw[0]['Rs'], lenstronomy_params_coretnfw[0]['Rs'])
        npt.assert_almost_equal(lenstronomy_params_tnfw[0]['r_trunc'], lenstronomy_params_coretnfw[0]['r_trunc'])


if __name__ == '__main__':
    pytest.main()
