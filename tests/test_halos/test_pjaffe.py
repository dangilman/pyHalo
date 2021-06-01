import numpy.testing as npt
import numpy as np
from pyHalo.Halos.HaloModels.PsuedoJaffe import PJaffeSubhalo
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Cosmology.cosmology import Cosmology
from colossus.halo.concentration import concentration
import pytest
from lenstronomy.LensModel.Profiles.p_jaffe import PJaffe

class TestPjaffeHalo(object):

    def setup(self):

        mass = 10**8.
        x = 0.5
        y = 1.
        r3d = np.sqrt(1 + 0.5 ** 2 + 70**2)
        self.r3d = r3d
        mdef = 'PJAFFE'
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
                        'log_mc': 0., 'c_scale': 60.,
                        'c_power': -0.17, 'c_scatter': False,
                        'mc_model': 'diemer19', 'LOS_truncation_factor': 40,
                        'mc_mdef': '200c',
                        'c_scatter_dex': 0.1}

        self.profile_args = profile_args

        self.mass_subhalo = mass
        self.subhalo = PJaffeSubhalo(mass, x, y, r3d, mdef, self.z,
                            sub_flag, self.lens_cosmo,
                            profile_args, unique_tag=np.random.rand())

        mass_field_halo = 10 ** 7.
        sub_flag = False
        self.mass_field_halo = mass_field_halo
        self.field_halo = PJaffeSubhalo(self.mass_field_halo, x, y, r3d, mdef, self.z,
                            sub_flag, self.lens_cosmo,
                            profile_args, unique_tag=np.random.rand())

        self._model_lenstronomy = PJaffe()


    def test_lenstronomy_ID(self):
        id = self.subhalo.lenstronomy_ID
        npt.assert_string_equal(id[0], 'PJAFFE')

        id = self.field_halo.lenstronomy_ID
        npt.assert_string_equal(id[0], 'PJAFFE')

    def test_z_infall(self):

        z_infall = self.subhalo.z_infall
        npt.assert_equal(True, self.z <= z_infall)

    def test_total_mass(self):

        c = float(self.subhalo.profile_args)
        rhos, rs, r200 = self.lens_cosmo.NFW_params_physical(self.subhalo.mass, c, self.z)
        fc = np.log(1 + c) - c / (1 + c)
        m_nfw = 4 * np.pi * rs ** 3 * rhos * fc

        lenstronomy_kwargs, _ = self.subhalo.lenstronomy_params
        sigma0, ra, rs = lenstronomy_kwargs['sigma0'], lenstronomy_kwargs['Ra'], lenstronomy_kwargs['Rs']

        arcsec_to_kpc = self.lens_cosmo.cosmo.kpc_proper_per_asec(self.z)
        ra *= arcsec_to_kpc ** -1
        rs *= arcsec_to_kpc ** -1
        rho = self.subhalo._rho(m_nfw, rs, ra, c*rs)

        m3d = self._model_lenstronomy.mass_3d(c*rs, rho, ra, rs)
        npt.assert_almost_equal(np.log10(m3d), np.log10(m_nfw))

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


if __name__ == '__main__':
    pytest.main()

