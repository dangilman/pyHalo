import numpy as np
from pyHalo.Scattering.isothermal_jeans import compute_r1, integrate_profile, solve_iterative
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Halos.lens_cosmo import LensCosmo
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter

cosmo = Cosmology()
lens_cosmo = LensCosmo(0.5, 1.5, cosmo)

def TNFWprofile(r, rhos, rs, rt):

    return SIDMprofileApprox(r, rhos, rs, rt, 0.)

def SIDMprofileApprox(r, rhos, rs, rt, rc, a=10):

    x = r / rs
    tau = rt / rs
    beta = rc/rs

    truncation_factor = tau ** 2 / (tau ** 2 + x ** 2)
    core_factor = (x ** a + beta ** a) ** (-1/a)

    return rhos * truncation_factor * core_factor * (1 + x) ** -2

class CompositeSIDMProfile(object):

    def __init__(self, M, z, cross_section_norm, v_power, N_solve=5, plot=False,
                 rmax_fac=10, rmin_fac=0.001, x_min=None, x_max=None):

        thalo = cosmo.halo_age(z)

        rhonfw, rs_nfw, _ = lens_cosmo.NFW_params_physical_fromM(M, z)

        rho0, s0, core_size_unitsrs, fit_quality, keywords = \
            solve_iterative(rhonfw, rs_nfw, cross_section_norm,
                            v_power, thalo, N_solve, plot,
                            rmin_fac, rmax_fac)


        r_1 = compute_r1(rhonfw, rs_nfw, s0, cross_section_norm, v_power, thalo)

        if x_max is None or x_max is None:
            r, rho_isothermal = integrate_profile(rho0, s0, rs_nfw, r_1,
                                              rmin_fac=rmin_fac, rmax_fac=rmax_fac)
        else:

            r_min, r_max = x_min * rs_nfw, x_max * rs_nfw

            r, rho_isothermal = integrate_profile(rho0, s0, rs_nfw, r_1,
                                              r_min=r_min, r_max=r_max)

        self.keywords_profile = keywords

        self.rho0 = rho0
        self.r_1 = r_1
        self.velocity_dispersion = s0
        self.rcore_units_kpc = rs_nfw * core_size_unitsrs
        self.rc_over_rs = core_size_unitsrs

        self.r_iso = r
        self.rho_isothermal = rho_isothermal
        self.rmax = r[-1]
        self.rmin = r[0]
        self._rmatch = core_size_unitsrs * rs_nfw

        assert self.rmin < self._rmatch

        if self.rmax <= self.r_1:
            print(self.rmax)
            print(self.r_1)
            raise Exception('must choose a larger value of rmax_fac')

        self.rho_iso_interp = interp1d(r, rho_isothermal)

        self.rhos, self.rs = rhonfw, rs_nfw

    def __call__(self, r, rt=100000, smooth=False, smooth_scale=0.01):


        if isinstance(r, np.ndarray) or isinstance(r, list):
            out = np.empty_like(r)

            inds_0 = np.where(r <= self.rmin)
            inds_iso = np.where((r > self.rmin) & (r <= self._rmatch))
            inds_nfw = np.where(r > self._rmatch)

            out[inds_0] = 0.
            out[inds_iso] = self.rho_iso_interp(r[inds_iso])
            out[inds_nfw] = TNFWprofile(r[inds_nfw], self.rhos, self.rs, rt)

            if smooth:
                out = gaussian_filter(out, sigma=smooth_scale*self.rs)
            return out

        else:

            if r <= self.rmin:
                return 0.
            elif r <= self._rmatch:
                return self.rho_iso_interp(r)
            else:
                return TNFWprofile(r, self.rhos, self.rs, rt)


