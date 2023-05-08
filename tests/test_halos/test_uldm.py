from pyHalo.Halos.concentration import ConcentrationDiemerJoyce
import numpy.testing as npt
import numpy as np
from pyHalo.Halos.HaloModels.ULDM import ULDMFieldHalo, ULDMSubhalo
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Cosmology.cosmology import Cosmology
from lenstronomy.LensModel.Profiles.cnfw import CNFW
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.LensModel.Profiles.uldm import Uldm
import pytest


class TestULDMHalo(object):

    def setup_method(self):
        astropy = FlatLambdaCDM(70.0, 0.3)
        cosmo = Cosmology(astropy_instance=astropy)
        self.zhalo = 0.5
        self.zsource = 2.0
        self.lens_cosmo = LensCosmo(self.zhalo, self.zsource, cosmo)
        self.truncation_class = None
        self.concentration_class = ConcentrationDiemerJoyce(self.lens_cosmo, scatter=False)

        self.mass = 10 ** 8
        x = 0.0
        y = 0.0
        r3d = 100.0
        self.kwargs_profile = {'evaluate_mc_at_zlens': False,
                          'log10_m_uldm': -21, 'uldm_plaw': 1 / 3, 'scale_nfw': True}
        self.subhalo = ULDMSubhalo(self.mass, x, y, r3d, self.zhalo,
                                   True, self.lens_cosmo,
                                   self.kwargs_profile, self.truncation_class, self.concentration_class,
                       unique_tag=np.random.rand())
        self.fieldhalo = ULDMFieldHalo(self.mass, x, y, r3d, self.zhalo,
                                       False, self.lens_cosmo,
                                       self.kwargs_profile, self.truncation_class, self.concentration_class,
                                       unique_tag=np.random.rand())

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
        npt.assert_equal(z_field, self.zhalo)
        # because the concentration is evaluated at infall, and z_infall > z
        npt.assert_equal(True, z_subhalo > z_field)

    def test_profile_normalization_subhalo(self):
        """
        Test that the mass enclosed within r200 of the composite profile is correct
        and check that the ULDM core density is correct.
        """

        [cnfw_kwargs, uldm_kwargs] = self.subhalo.lenstronomy_params[0]
        Rs_angle, _ = self.lens_cosmo.nfw_physical2angle(10**8, self.subhalo.c, self.subhalo.z)
        sigma_crit = self.lens_cosmo.sigmacrit
        r200 = self.subhalo.c * Rs_angle

        M_nfw = CNFW().mass_3d_lens(r200, cnfw_kwargs['Rs'], cnfw_kwargs['alpha_Rs']*sigma_crit, cnfw_kwargs['r_core'])
        M_uldm = Uldm().mass_3d_lens(r200, uldm_kwargs['kappa_0']*sigma_crit, uldm_kwargs['theta_c'])

        npt.assert_almost_equal((M_uldm+M_nfw)/self.subhalo.mass,1,decimal=2) # less than 1% error
        _,theta_c,kappa_0 = self.subhalo.profile_args
        rho0 = Uldm().density_lens(0,uldm_kwargs['kappa_0'],
                                    uldm_kwargs['theta_c'])
        rhos = CNFW().density_lens(0,cnfw_kwargs['Rs'],
                                 cnfw_kwargs['alpha_Rs'],
                                 cnfw_kwargs['r_core'])
        rho_goal = Uldm().density_lens(0,kappa_0,theta_c)
        npt.assert_array_less(np.array([1-(rho0+rhos)/rho_goal]),np.array([0.1])) # less than 3% error

    def test_profile_normalization_fieldhalo(self):
        """
        Test that the mass enclosed within r200 of the composite profile is correct
        and check that the ULDM core density is correct.
        """

        [cnfw_kwargs, uldm_kwargs] = self.fieldhalo.lenstronomy_params[0]
        Rs_angle, _ = self.lens_cosmo.nfw_physical2angle(self.mass, self.fieldhalo.c, self.fieldhalo.z)
        sigma_crit = self.lens_cosmo.sigmacrit
        r200 = self.fieldhalo.c * Rs_angle

        M_nfw = CNFW().mass_3d_lens(r200, cnfw_kwargs['Rs'], cnfw_kwargs['alpha_Rs'] * sigma_crit,
                                    cnfw_kwargs['r_core'])
        M_uldm = Uldm().mass_3d_lens(r200, uldm_kwargs['kappa_0'] * sigma_crit, uldm_kwargs['theta_c'])

        npt.assert_almost_equal((M_uldm + M_nfw) / self.fieldhalo.mass, 1, decimal=2)  # less than 1% error
        _, theta_c, kappa_0 = self.fieldhalo.profile_args
        rho0 = Uldm().density_lens(0, uldm_kwargs['kappa_0'],
                                   uldm_kwargs['theta_c'])
        rhos = CNFW().density_lens(0, cnfw_kwargs['Rs'],
                                   cnfw_kwargs['alpha_Rs'],
                                   cnfw_kwargs['r_core'])
        rho_goal = Uldm().density_lens(0, kappa_0, theta_c)
        npt.assert_array_less(np.array([1 - (rho0 + rhos) / rho_goal]), np.array([0.1]))  # less than 10% error


if __name__ == '__main__':
   pytest.main()
