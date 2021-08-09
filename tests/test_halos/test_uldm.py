from astropy.cosmology.funcs import z_at_value
import numpy as np
from pyHalo.single_realization import SingleHalo
import numpy.testing as npt
import numpy as np
from pyHalo.Halos.HaloModels.ULDM import ULDMFieldHalo, ULDMSubhalo
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Cosmology.cosmology import Cosmology
from lenstronomy.LensModel.Profiles.cnfw import CNFW
from lenstronomy.LensModel.Profiles.nfw import NFW
from lenstronomy.LensModel.Profiles.uldm import Uldm 
import pytest

class TestULDMHalo(object):

    def setup(self):

        mass = 1e9
        x = 0.5
        y = 1.
        r3d = np.sqrt(1 + 0.5 ** 2 + 70**2)
        self.r3d = r3d
        self.z = 0.25
        sub_flag = True
        mdef = 'ULDM'
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
                        'evaluate_mc_at_zlens': False,
                        'log_mc': None, 'c_scale': 60.,
                        'c_power': -0.17, 'c_scatter': False,
                        'mc_model': 'diemer19', 'LOS_truncation_factor': 40,
                        'c_scatter_dex': 0.1, 'mc_mdef': '200c',
                        'log10_m_uldm':-22, 'uldm_plaw':1/3}

        self.subhalo = ULDMSubhalo(mass, x, y, r3d, mdef, self.z,
                                   sub_flag, self.lens_cosmo,
                                   profile_args, unique_tag=np.random.rand())
        self.fieldhalo = ULDMFieldHalo(mass, x, y, r3d, mdef, self.z,
                                       sub_flag, self.lens_cosmo,
                                       profile_args, unique_tag=np.random.rand())

    def test_lenstronomy_ID(self):

        ID = self.fieldhalo.lenstronomy_ID
        npt.assert_string_equal(ID[0], 'CNFW')
        npt.assert_string_equal(ID[1], 'ULDM')

        ID = self.subhalo.lenstronomy_ID
        npt.assert_string_equal(ID[0], 'CNFW')
        npt.assert_string_equal(ID[1], 'ULDM')

    def test_redshift_eval(self):

        z_subhalo = self.subhalo.z_eval
        z_field = self.fieldhalo.z_eval
        npt.assert_equal(z_field, self.z)
        # because the concentration is evaluated at infall, and z_infall > z
        npt.assert_equal(True, z_subhalo > z_field)

    def test_profile_load(self):

        # test cored composite profile

        profile_args = {'log10_m_uldm': -22, 'uldm_plaw': 1/3, 'scale_nfw':False}

        single_halo = SingleHalo(1e8, 0.5, 0.5, 'ULDM', 0.5, 0.5, 1.5, None, True, profile_args, None)
        lens_model_list, redshift_array, kwargs_lens, numerical_interp = single_halo.\
            lensing_quantities(add_mass_sheet_correction=False)
        npt.assert_string_equal(lens_model_list[1], 'ULDM')
        npt.assert_string_equal(lens_model_list[0], 'CNFW')
        npt.assert_equal(True, len(kwargs_lens)==2)
        npt.assert_equal(True, len(redshift_array)==2)

    def test_profile_normalization(self):
        """
        Test that the mass enclosed within r200 of the composite profile is correct 
        and check that the ULDM core density is correct.
        """
        profile_args = {'log10_m_uldm': -21, 'uldm_plaw': 1/3, 'scale_nfw':True}
        mass = 1e10
        zl = 0.5
        zs = 1.5
        single_halo = SingleHalo(mass, 0.5, 0.5, 'ULDM', zl, zl, zs, None, True, profile_args, None)
        _, _, kwargs_lens, _ = single_halo.lensing_quantities(add_mass_sheet_correction=False)
        Rs_angle, _ = single_halo.halos[0].lens_cosmo.nfw_physical2angle(mass, single_halo.halos[0].c, zl)
        sigma_crit = single_halo.halos[0].lens_cosmo.sigmacrit
        r200 = single_halo.halos[0].c * Rs_angle
        cnfw_kwargs, uldm_kwargs = kwargs_lens
        M_nfw = CNFW().mass_3d_lens(r200, cnfw_kwargs['Rs'], cnfw_kwargs['alpha_Rs']*sigma_crit, cnfw_kwargs['r_core'])
        M_uldm = Uldm().mass_3d_lens(r200, uldm_kwargs['kappa_0']*sigma_crit, uldm_kwargs['theta_c'])
        npt.assert_almost_equal((M_uldm+M_nfw)/mass,1,decimal=2) # less than 1% error
        _,theta_c,kappa_0 = single_halo.halos[0].profile_args
        rho0 = Uldm().density_lens(0,uldm_kwargs['kappa_0'],
                                    uldm_kwargs['theta_c'])
        rhos = CNFW().density_lens(0,cnfw_kwargs['Rs'],
                                 cnfw_kwargs['alpha_Rs'],
                                 cnfw_kwargs['r_core'])
        rho_goal = Uldm().density_lens(0,kappa_0,theta_c)
        npt.assert_array_less(np.array([1-(rho0+rhos)/rho_goal]),np.array([0.02])) # less than 2% error

if __name__ == '__main__':
   pytest.main()
