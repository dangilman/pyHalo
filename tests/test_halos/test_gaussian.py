import numpy.testing as npt
import numpy as np
from pyHalo.Halos.HaloModels.gaussian import Gaussian
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.single_realization import SingleHalo
from lenstronomy.LensModel.Profiles.gaussian_kappa import GaussianKappa

class TestGaussianHalo(object):

    def setup(self):

        mass = 10 ** 8.
        x = 0.5
        y = 1.
        z = 0.9
        r3d = np.sqrt(1 + 0.5 ** 2 + 70 ** 2)
        self.r3d = r3d
        mdef = 'GAUSSIAN_KAPPA'
        self.z = z
        sub_flag = True

        self.H0 = 70
        self.omega_baryon = 0.03
        self.omega_DM = 0.25
        self.sigma8 = 0.82
        curvature = 'flat'
        self.ns = 0.9608
        cosmo_params = {'H0': self.H0, 'Om0': self.omega_baryon + self.omega_DM, 'Ob0': self.omega_baryon,
                        'sigma8': self.sigma8, 'ns': self.ns, 'curvature': curvature}

        cosmo = Cosmology(cosmo_kwargs=cosmo_params)
        self.lens_cosmo = LensCosmo(self.z, 2., cosmo)

        profile_args = {'amp':1,'sigma':1,'center_x':0,'center_y':0}

        sub_flag = False
        self.halo = Gaussian(mass, x, y, r3d, mdef,z,
                               sub_flag, self.lens_cosmo,
                               profile_args, unique_tag=np.random.rand())

    def test_lenstronomy_ID(self):

        id = self.halo.lenstronomy_ID
        npt.assert_string_equal(id[0], 'GAUSSIAN_KAPPA')

    def test_redshift_eval(self):

        z_halo = self.halo.z_eval
        npt.assert_equal(z_halo, self.z)

    def test_profile_load(self):

        profile_args = {'amp':1,'sigma':1,'center_x':0,'center_y':0}

        single_halo = SingleHalo(1e8, 0.5, 0.5, 'GAUSSIAN_KAPPA', 0.5, 0.5, 1.5, None, True, profile_args, None)
        lens_model_list, redshift_array, kwargs_lens, numerical_interp = single_halo.\
            lensing_quantities(add_mass_sheet_correction=False)
        npt.assert_string_equal(lens_model_list[0], 'GAUSSIAN_KAPPA')

    def test_mass(self):

        'Check that mass definition of 5sigma'

        mass = 1
        profile_args = {'amp':0.1,'sigma':0.1,'center_x':0,'center_y':0}
        single_halo = SingleHalo(mass, 0.5, 0.5, 'GAUSSIAN_KAPPA', 0.5, 0.5, 1.5, None, True, profile_args, None)
        lens_model_list, redshift_array, kwargs_lens, numerical_interp = single_halo.\
            lensing_quantities(add_mass_sheet_correction=False)

        M_trial=GaussianKappa().mass_3d_lens(5*kwargs_lens[0]['sigma'],kwargs_lens[0]['amp'],kwargs_lens[0]['sigma'])
        npt.assert_almost_equal(np.abs(mass-M_trial),0,decimal=5)

if __name__ == '__main__':
   pytest.main()


