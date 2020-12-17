import numpy.testing as npt
import numpy as np
from pyHalo.Halos.halo import Halo
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Cosmology.cosmology import Cosmology
from colossus.halo.concentration import concentration, peaks
import pytest


class TestNFWHalos(object):

    def setup(self):

        mass = 10**8.
        x = 0.5
        y = 1.
        r2d = np.sqrt(x**2 + y**2)
        r3d = np.sqrt(r2d**2 + 70**2)
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
                        'log_m_break': 0., 'c_scale': 60.,
                        'c_power': -0.17, 'c_scatter': False,
                        'mc_model': 'diemer19', 'LOS_truncation_factor': 40}

        self._profile_args = profile_args

        self.mass_subhalo = mass
        self.subhalo = Halo(mass, x, y, r2d, r3d, mdef, self.z,
                            sub_flag, self.lens_cosmo,
                            profile_args, unique_tag=None)

        mass_field_halo = 10 ** 7.
        sub_flag = False
        self.mass_field_halo = mass_field_halo
        self.field_halo = Halo(self.mass_field_halo, x, y, r2d, r3d, mdef, self.z,
                            sub_flag, self.lens_cosmo,
                            profile_args, unique_tag=None)

        self.profile_args_custom = {'RocheNorm': 1.2, 'RocheNu': 2/3,
                        'evaluate_mc_at_zlens': True,
                        'log_m_break': 0., 'c_scale': 60.,
                        'c_power': -0.17, 'c_scatter': False,
                        'mc_model': {'custom': True, 'c0': 17., 'beta': 0.8, 'zeta': -0.2},
                               'LOS_truncation_factor': 40}

        mdef = 'NFW'
        sub_flag = False
        self.field_halo_custom = Halo(self.mass_field_halo, x, y, r2d, r3d, mdef, self.z,
                               sub_flag, self.lens_cosmo,
                               self.profile_args_custom, unique_tag=None)

        sub_flag = True
        self.subhalo_custom = Halo(self.mass_subhalo, x, y, r2d, r3d, mdef, self.z,
                                      sub_flag, self.lens_cosmo,
                                      self.profile_args_custom, unique_tag=None)


    def test_change_profile_definition(self):

        new_mdef = 'PT_MASS'

        new_halo = Halo.change_profile_definition(self.subhalo, new_mdef)
        npt.assert_almost_equal(new_halo.x, self.subhalo.x)
        npt.assert_almost_equal(new_halo.y, self.subhalo.y)
        npt.assert_almost_equal(new_halo.r2d, self.subhalo.r2d)
        npt.assert_almost_equal(new_halo.r3d, self.subhalo.r3d)
        npt.assert_almost_equal(new_halo.mass, self.subhalo.mass)
        npt.assert_almost_equal(new_halo._unique_tag, self.subhalo._unique_tag)
        npt.assert_string_equal(new_halo.mdef, new_mdef)

    def test_z_infall(self):

        z_infall = self.subhalo.get_z_infall()
        npt.assert_array_less(self.z, z_infall)

    def profile_args(self):

        profile_args = self.subhalo.profile_args
        (c, rt) = profile_args
        con = concentration(self.lens_cosmo.cosmo.h * self.mass_subhalo, '200c', self.z,
                            model='diemer19')
        npt.assert_almost_equal(c/con, 1, 2)
        trunc = self._profile_args['RocheNorm'] * (10 ** 8 / 10 ** 7) ** (1./3) * \
                (self.r3d / 50) ** self._profile_args['RocheNu']
        npt.assert_almost_equal(trunc, rt, 3)

        profile_args = self.field_halo.profile_args
        (c, rt) = profile_args
        con = concentration(self.lens_cosmo.cosmo.h * self.mass_field_halo, '200c', self.z,
                            model='diemer19')
        npt.assert_almost_equal(c / con, 1, 2)
        m_h = self.mass_field_halo * self.lens_cosmo.cosmo.h
        r50_comoving = self.lens_cosmo.rN_M_nfw_comoving(m_h, self._profile_args['LOS_truncation_factor'], self.z)
        r50_physical = r50_comoving * self.lens_cosmo.cosmo.scale_factor(self.z) / self.lens_cosmo.cosmo.h
        r50_physical_kpc = r50_physical * 1000
        npt.assert_almost_equal(r50_physical_kpc, rt, 3)

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
        npt.assert_almost_equal(con_subhalo/c, 1, 2)

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
        npt.assert_almost_equal(con_field_halo / c, 1, 2)

if __name__ == '__main__':
    pytest.main()

