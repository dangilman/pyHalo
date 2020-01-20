import numpy.testing as npt
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Halos.halo_util import *
from pyHalo.Halos.lens_cosmo import LensCosmo
import numpy as np
import pytest

class TestStructuralParameters(object):

    def setup(self):

        H0 = 70
        omega_baryon = 0.03
        omega_DM = 0.25
        sigma8 = 0.82
        curvature = 'flat'
        ns = 0.9608
        cosmo_params = {'H0': H0, 'Om0': omega_baryon + omega_DM, 'Ob0': omega_baryon,
                      'sigma8': sigma8, 'ns': ns, 'curvature': curvature}
        self._dm, self._bar = omega_DM, omega_baryon
        self.cosmo = Cosmology(cosmo_kwargs=cosmo_params)

        zlens, zsource = 0.5, 1.5
        self.lens_cosmo = LensCosmo(zlens, zsource, self.cosmo)

        self.sigmacrit = self.lens_cosmo.sigmacrit

        self.rs, self.rho_s = 40, 10 ** 6
        self.rmax = 5 * self.rs
        self.r_ein_kpc = 8

    def test_numerical_integrals(self):

        def dummy_profile(_r, _rs, _norm):
            _x = _r/_rs
            return _norm / (1+_x**2)

        density_function = dummy_profile
        profile_args = (1, 1)
        zmax = 150000

        mean_kappa_numerical = mean_kappa(4, density_function, profile_args, zmax)
        true_value_mathematica = 1.22644
        npt.assert_almost_equal(true_value_mathematica/mean_kappa_numerical, 1, 5)

    def test_NFW(self):

        density_profile = rho_nfw
        args = (self.rs, self.rho_s)

        mass_numerical = quad(mass_integral_numerical, 0, self.rmax, (density_profile, args))[0]
        mass_analytical = nfw_profile_mass(self.rmax, self.rs, self.rho_s)
        npt.assert_almost_equal(mass_analytical / mass_numerical, 1, 2)

        numerical_mean_density = mean_density_numerical(self.rmax, density_profile, args)
        analytic_mean_density = nfw_profile_mean_density(self.rmax, self.rs, self.rho_s)

        npt.assert_almost_equal(numerical_mean_density/analytic_mean_density, 1, 2)

    def test_composite(self):

        density_profile = rho_composite
        rho0 = density_norm(self.rs, self.r_ein_kpc, self.sigmacrit)
        args = (self.rs, rho0)

        mass_numerical = quad(mass_integral_numerical, 0, self.rmax, (density_profile, args))[0]
        mass_analytical = composite_profile_mass(self.rmax, self.rs, rho0)
        npt.assert_almost_equal(mass_analytical/mass_numerical, 1, 5)

        numerical_mean_density = mean_density_numerical(self.rmax, density_profile, args)
        analytic_mean_density = composite_profile_mean_density(self.rmax, self.rs, rho0)
        npt.assert_almost_equal(numerical_mean_density / analytic_mean_density, 1, 2)

        mean_kappa_numerical = mean_kappa(self.r_ein_kpc, density_profile, args, 20000*self.rs)
        npt.assert_almost_equal(mean_kappa_numerical/self.sigmacrit, 1, 3)


if __name__ == '__main__':
    pytest.main()
