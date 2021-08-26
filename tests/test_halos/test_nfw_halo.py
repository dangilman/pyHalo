import numpy.testing as npt
import numpy as np
from pyHalo.Halos.HaloModels.NFW import NFWSubhhalo, NFWFieldHalo
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Cosmology.cosmology import Cosmology
from colossus.halo.concentration import concentration, peaks
import pytest

class TestNFWHalos(object):

    def setup(self):

        mass = 10**8.
        x = 0.5
        y = 1.
        r3d = np.sqrt(1 + 0.5 ** 2 + 70**2)
        self.r3d = r3d
        mdef = 'TNFW'
        self.z = 1.2
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
                        'mc_mdef': '200c',
                        'c_scatter_dex': 0.1}

        self._profile_args = profile_args

        self.mass_subhalo = mass
        self.subhalo = NFWSubhhalo(mass, x, y, r3d, mdef, self.z,
                            sub_flag, self.lens_cosmo,
                            profile_args, unique_tag=np.random.rand())

        mass_field_halo = 10 ** 7.
        sub_flag = False
        self.mass_field_halo = mass_field_halo
        self.field_halo = NFWFieldHalo(self.mass_field_halo, x, y, r3d, mdef, self.z,
                            sub_flag, self.lens_cosmo,
                            profile_args, unique_tag=np.random.rand())

        self.profile_args_custom = {'RocheNorm': 1.2, 'RocheNu': 2/3,
                        'evaluate_mc_at_zlens': True,
                        'log_mc': None, 'c_scale': 60.,
                        'c_power': -0.17, 'c_scatter': False,
                        'mc_model': {'custom': True, 'c0': 28., 'beta': 1.2, 'zeta': -0.5},
                               'LOS_truncation_factor': 40, 'mc_mdef': '200c',
                                    'c_scatter_dex': 0.1}

        mdef = 'NFW'
        sub_flag = False
        self.field_halo_custom = NFWFieldHalo(self.mass_field_halo, x, y, r3d, mdef, self.z,
                               sub_flag, self.lens_cosmo,
                               self.profile_args_custom, unique_tag=np.random.rand())

        sub_flag = True
        self.subhalo_custom = NFWSubhhalo(self.mass_subhalo, x, y, r3d, mdef, self.z,
                                      sub_flag, self.lens_cosmo,
                                      self.profile_args_custom, unique_tag=np.random.rand())


    def test_lenstronomy_ID(self):

        id = self.subhalo_custom.lenstronomy_ID
        npt.assert_string_equal(id[0], 'NFW')

    def test_z_infall(self):

        z_infall = self.subhalo.z_infall
        npt.assert_equal(True, self.z <= z_infall)

    def test_lenstronomy_kwargs(self):

        for prof in [self.subhalo_custom, self.subhalo, self.field_halo, self.field_halo_custom]:

            (c) = prof.profile_args
            kwargs, other = prof.lenstronomy_params
            rs, theta_rs = self.lens_cosmo.nfw_physical2angle(prof.mass, c, self.z)
            names = ['center_x', 'center_y', 'alpha_Rs', 'Rs']
            values = [prof.x, prof.y, theta_rs, rs]
            for name, value in zip(names, values):
                npt.assert_almost_equal(kwargs[0][name]/value, 1, 5)

    def test_profile_args(self):

        profile_args = self.subhalo.profile_args

        (c) = profile_args
        con = concentration(self.lens_cosmo.cosmo.h * self.mass_subhalo, '200c', self.z,
                            model='diemer19')
        npt.assert_almost_equal(c/con, 1, 2)

        profile_args = self.field_halo.profile_args
        (c) = profile_args
        con = concentration(self.lens_cosmo.cosmo.h * self.mass_field_halo, '200c', self.z,
                            model='diemer19')
        npt.assert_almost_equal(c / con, 1, 2)

        profile_args = self.subhalo_custom.profile_args
        (c) = profile_args
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
        (c) = profile_args
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

