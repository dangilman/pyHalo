import numpy.testing as npt
import numpy as np
from pyHalo.Halos.HaloModels.TNFW import TNFWFieldHalo, TNFWSubhhalo
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Cosmology.cosmology import Cosmology
from colossus.halo.concentration import concentration, peaks
import pytest


class TestTNFWHalos(object):

    def setup(self):

        mass = 10**8.
        x = 0.5
        y = 1.
        r3d = np.sqrt(1 + 0.5 ** 2 + 70**2)
        self.r3d = r3d
        mdef = 'TNFW'
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

        profile_args = {'RocheNorm': 1.2, 'RocheNu': 2/3,
                        'evaluate_mc_at_zlens': True,
                        'log_mc': None, 'c_scale': 60.,
                        'c_power': -0.17, 'c_scatter': False,
                        'mc_model': 'diemer19', 'LOS_truncation_factor': 40,
                        'c_scatter_dex': 0.1, 'mc_mdef': '200c'}

        self.profile_args_WDM = {'RocheNorm': 1.2, 'RocheNu': 2 / 3,
                        'evaluate_mc_at_zlens': True,
                        'log_mc': 7.6, 'c_scale': 60.,
                        'c_power': -0.17, 'c_scatter': False,
                        'mc_model': 'diemer19', 'LOS_truncation_factor': 40,
                        'c_scatter_dex': 0.1, 'mc_mdef': '200c'}

        self._profile_args = profile_args

        self.mass_subhalo = mass
        self.subhalo = TNFWSubhhalo(mass, x, y, r3d, mdef, self.z,
                            sub_flag, self.lens_cosmo,
                            profile_args, unique_tag=np.random.rand())
        self.subhalo_WDM = TNFWSubhhalo(mass, x, y, r3d, mdef, self.z,
                                    sub_flag, self.lens_cosmo,
                                    self.profile_args_WDM, unique_tag=np.random.rand())

        mass_field_halo = 10 ** 7.
        sub_flag = False
        self.mass_field_halo = mass_field_halo
        self.field_halo = TNFWFieldHalo(self.mass_field_halo, x, y, r3d, mdef, self.z,
                            sub_flag, self.lens_cosmo,
                            profile_args, unique_tag=np.random.rand())
        self.field_halo_WDM = TNFWFieldHalo(self.mass_field_halo, x, y, r3d, mdef, self.z,
                                        sub_flag, self.lens_cosmo,
                                        self.profile_args_WDM, unique_tag=np.random.rand())

        self.profile_args_custom = {'RocheNorm': 1.2, 'RocheNu': 2/3,
                        'evaluate_mc_at_zlens': True,
                        'log_mc': None, 'c_scale': 60.,
                        'c_power': -0.17, 'c_scatter': False,
                        'mc_model': {'custom': True, 'c0': 6., 'beta': 0.2, 'zeta': -0.3},
                               'LOS_truncation_factor': 40,
                                    'c_scatter_dex': 0.1, 'mc_mdef': '200c'}

        mdef = 'NFW'
        sub_flag = False
        self.field_halo_custom = TNFWFieldHalo(self.mass_field_halo, x, y, r3d, mdef, self.z,
                               sub_flag, self.lens_cosmo,
                               self.profile_args_custom, unique_tag=np.random.rand())

        sub_flag = True
        self.subhalo_custom = TNFWSubhhalo(self.mass_subhalo, x, y, r3d, mdef, self.z,
                                      sub_flag, self.lens_cosmo,
                                      self.profile_args_custom, unique_tag=np.random.rand())

        self.profile_args_WDM_custom = {'RocheNorm': 1.2, 'RocheNu': 2 / 3,
                                    'evaluate_mc_at_zlens': True,
                                    'log_mc': 8., 'c_scale': 40.,
                                    'c_power': -0.3, 'c_scatter': False,
                                    'mc_model': {'custom': True, 'c0': 6., 'beta': 0.2, 'zeta': -0.3},
                                    'LOS_truncation_factor': 40,
                                    'c_scatter_dex': 0.1, 'mc_mdef': '200c'}
        sub_flag = True
        self.subhalo_custom_WDM = TNFWSubhhalo(self.mass_subhalo, x, y, r3d, mdef, self.z,
                                           sub_flag, self.lens_cosmo,
                                           self.profile_args_WDM_custom, unique_tag=np.random.rand())
        sub_flag = False
        self.field_halo_custom_WDM = TNFWFieldHalo(self.mass_field_halo, x, y, r3d, mdef, self.z,
                                               sub_flag, self.lens_cosmo,
                                               self.profile_args_WDM_custom, unique_tag=np.random.rand())


    def test_lenstronomy_kwargs(self):

        for prof in [self.subhalo_custom, self.subhalo, self.field_halo, self.field_halo_custom]:

            (c, rtrunc_kpc) = prof.profile_args
            kwargs, other = prof.lenstronomy_params
            rs, theta_rs = self.lens_cosmo.nfw_physical2angle(prof.mass, c, self.z)
            names = ['center_x', 'center_y', 'alpha_Rs', 'Rs', 'r_trunc']
            rtrunc_angle = rtrunc_kpc / self.lens_cosmo.cosmo.kpc_proper_per_asec(self.z)
            values = [prof.x, prof.y, theta_rs, rs, rtrunc_angle]
            for name, value in zip(names, values):
                npt.assert_almost_equal(kwargs[name], value)

    def test_lenstronomy_ID(self):
        id = self.subhalo_custom.lenstronomy_ID
        npt.assert_string_equal(id, 'TNFW')

    def test_change_profile_definition(self):

        new_mdef = 'PT_MASS'

        new_halo = TNFWSubhhalo.change_profile_definition(self.subhalo, new_mdef)
        npt.assert_almost_equal(new_halo.x, self.subhalo.x)
        npt.assert_almost_equal(new_halo.y, self.subhalo.y)
        npt.assert_almost_equal(new_halo.r3d, self.subhalo.r3d)
        npt.assert_almost_equal(new_halo.mass, self.subhalo.mass)
        npt.assert_almost_equal(new_halo.unique_tag, self.subhalo.unique_tag)
        npt.assert_string_equal(new_halo.mdef, new_mdef)

        new_halo = TNFWFieldHalo.change_profile_definition(self.field_halo, new_mdef)
        npt.assert_almost_equal(new_halo.x, self.field_halo.x)
        npt.assert_almost_equal(new_halo.y, self.field_halo.y)
        npt.assert_almost_equal(new_halo.r3d, self.field_halo.r3d)
        npt.assert_almost_equal(new_halo.mass, self.field_halo.mass)
        npt.assert_almost_equal(new_halo.unique_tag, self.field_halo.unique_tag)
        npt.assert_string_equal(new_halo.mdef, new_mdef)

    def test_z_infall(self):

        z_infall = self.subhalo.z_infall
        npt.assert_array_less(self.z, z_infall)

    def test_profile_args(self):

        profile_args = self.subhalo.profile_args
        (c, rt) = profile_args
        con = concentration(self.lens_cosmo.cosmo.h * self.mass_subhalo, '200c', self.z,
                            model='diemer19')
        npt.assert_almost_equal(c/con, 1, 2)
        trunc = self._profile_args['RocheNorm'] * (10 ** 8 / 10 ** 7) ** (1./3) * \
                (self.r3d / 50) ** self._profile_args['RocheNu']
        npt.assert_almost_equal(trunc, rt, 3)

        profile_args = self.subhalo_WDM.profile_args
        (c, rt) = profile_args
        con = concentration(self.lens_cosmo.cosmo.h * self.mass_subhalo, '200c', self.z,
                            model='diemer19')
        c_scale, c_power = self.profile_args_WDM['c_scale'], self.profile_args_WDM['c_power']
        wdm_suppresion = (1 + self.z) ** (0.026 * self.z - 0.04) * (1 + c_scale * 10 ** self.profile_args_WDM['log_mc'] /
                          self.subhalo.mass) ** c_power
        npt.assert_almost_equal(con * wdm_suppresion, c)

        profile_args = self.field_halo.profile_args
        (c, rt) = profile_args
        con = concentration(self.lens_cosmo.cosmo.h * self.mass_field_halo, '200c', self.z,
                            model='diemer19')
        npt.assert_almost_equal(c / con, 1, 2)
        m_h = self.mass_field_halo * self.lens_cosmo.cosmo.h
        r50_comoving = self.lens_cosmo.rN_M_nfw_comoving(m_h, self._profile_args['LOS_truncation_factor'], self.z)
        r50_physical = r50_comoving * self.lens_cosmo.cosmo.scale_factor(self.z) / self.lens_cosmo.cosmo.h
        r50_physical_kpc = r50_physical * 1000
        npt.assert_almost_equal(r50_physical_kpc, rt)

        profile_args = self.field_halo_WDM.profile_args
        (c, rt) = profile_args
        con = concentration(self.lens_cosmo.cosmo.h * self.mass_field_halo, '200c', self.z,
                            model='diemer19')
        c_scale, c_power = self.profile_args_WDM['c_scale'], self.profile_args_WDM['c_power']
        wdm_suppresion = (1 + self.z) ** (0.026 * self.z - 0.04) * (1 + c_scale * 10 ** self.profile_args_WDM['log_mc'] /
                          self.field_halo_WDM.mass) ** c_power
        npt.assert_almost_equal(con * wdm_suppresion, c)

        profile_args = self.subhalo_custom.profile_args
        (c, _) = profile_args
        c0 = self.profile_args_custom['mc_model']['c0']
        beta = self.profile_args_custom['mc_model']['beta']
        zeta = self.profile_args_custom['mc_model']['zeta']

        h = self.lens_cosmo.cosmo.h
        mh_sub = self.mass_subhalo * h
        nu = peaks.peakHeight(mh_sub, self.z)
        nu_ref = peaks.peakHeight(h * 10 ** 8, 0.)
        con_subhalo = c0 * (1 + self.z) ** zeta * (nu / nu_ref) ** -beta
        npt.assert_almost_equal(con_subhalo/c, 1)

        profile_args = self.field_halo_custom.profile_args
        (c, _) = profile_args
        c0 = self.profile_args_custom['mc_model']['c0']
        beta = self.profile_args_custom['mc_model']['beta']
        zeta = self.profile_args_custom['mc_model']['zeta']

        h = self.lens_cosmo.cosmo.h
        mh_sub = self.mass_field_halo * h
        nu = peaks.peakHeight(mh_sub, self.z)
        nu_ref = peaks.peakHeight(h * 10 ** 8, 0.)
        con_field_halo = c0 * (1 + self.z) ** zeta * (nu / nu_ref) ** -beta
        npt.assert_almost_equal(con_field_halo / c, 1)

        profile_args = self.field_halo_custom_WDM.profile_args
        (c, _) = profile_args
        c0 = self.profile_args_custom['mc_model']['c0']
        beta = self.profile_args_custom['mc_model']['beta']
        zeta = self.profile_args_custom['mc_model']['zeta']
        c_scale, c_power = self.profile_args_WDM_custom['c_scale'], self.profile_args_WDM_custom['c_power']
        wdm_suppresion = (1 + self.z) ** (0.026 * self.z - 0.04) * (1 +
                          c_scale * 10 ** self.profile_args_WDM_custom['log_mc'] /
                          self.field_halo_custom_WDM.mass) ** c_power

        h = self.lens_cosmo.cosmo.h
        mh_sub = self.mass_field_halo * h
        nu = peaks.peakHeight(mh_sub, self.z)
        nu_ref = peaks.peakHeight(h * 10 ** 8, 0.)
        con_field_halo = c0 * (1 + self.z) ** zeta * (nu / nu_ref) ** -beta
        npt.assert_almost_equal(wdm_suppresion * con_field_halo, c)


if __name__ == '__main__':
    pytest.main()

