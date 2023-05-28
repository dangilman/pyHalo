import pytest
import numpy.testing as npt
from pyHalo.Halos.concentration import ConcentrationDiemerJoyce
from pyHalo.Halos.tidal_truncation import AdiabaticTidesTruncation
from pyHalo.Halos.HaloModels.TNFW import TNFWSubhalo
from pyHalo.Halos.lens_cosmo import LensCosmo
import numpy as np

class DummyInterp(object):

    def __call__(self, x):
        #np.log10(c), np.log10(r_pericenter_over_r200), self._log10_galaxy_rs, self._log10_galaxy_m
        (log10c, log10_rperi_r200, chost) = x
        rperi_term = (10**log10_rperi_r200 / 1.0) ** 2
        c_term = (10**log10c / 500) **0.5
        return np.log10(c_term * rperi_term)

class TestAdiabaticTides(object):

    def setup_method(self):
        self.lens_cosmo_instance = LensCosmo(0.5, 1.5)
        self.att = AdiabaticTidesTruncation(self.lens_cosmo_instance,
                                            13.3, 0.5, mass_loss_interp=DummyInterp())


    def test_rt(self):

        mass = 10 ** 8.0
        x = 0.1
        y = 0.5
        r3d = None
        z = 0.5
        sub_flag = True
        kwargs_density_profile = {'evaluate_mc_at_zlens': True}
        truncation_class = self.att
        concentration_class = ConcentrationDiemerJoyce(self.lens_cosmo_instance.cosmo.astropy,
                                                       scatter=False)
        unique_tag = 1.0
        halo_1 = TNFWSubhalo(mass, x, y, r3d, z,
                 sub_flag, self.lens_cosmo_instance, kwargs_density_profile,
                 truncation_class, concentration_class, unique_tag)
        _ = halo_1.c
        halo_1._rperi_units_r200 = 0.4
        halo_1._time_since_infall = 1.0
        profile_params_1 = halo_1.lenstronomy_params[0]
        tau_1 = profile_params_1[0]['r_trunc']/profile_params_1[0]['Rs']

        halo_2 = TNFWSubhalo(mass, x, y, r3d, z,
                             sub_flag, self.lens_cosmo_instance, kwargs_density_profile,
                             truncation_class, concentration_class, unique_tag)
        _ = halo_2.c
        halo_2._rperi_units_r200 = 0.6
        halo_2._time_since_infall = 1.0
        profile_params_2 = halo_2.lenstronomy_params[0]
        tau_2 = profile_params_2[0]['r_trunc'] / profile_params_2[0]['Rs']

        npt.assert_equal(tau_2 > tau_1, True)

        halo_3 = TNFWSubhalo(mass, x, y, r3d, z,
                             sub_flag, self.lens_cosmo_instance, kwargs_density_profile,
                             truncation_class, concentration_class, unique_tag)
        _ = halo_3.c
        halo_3._c *= 5.0
        halo_3._rperi_units_r200 = 0.6
        halo_3._time_since_infall = 1.0
        profile_params_3 = halo_3.lenstronomy_params[0]
        tau_3 = profile_params_3[0]['r_trunc'] / profile_params_3[0]['Rs']
        npt.assert_equal(tau_3 > tau_2, True)

if __name__ == '__main__':
     pytest.main()

