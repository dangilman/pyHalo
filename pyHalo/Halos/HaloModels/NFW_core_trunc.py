from pyHalo.Halos.halo_base import Halo
from lenstronomy.LensModel.Profiles.nfw_core_truncated import TNFWC as TNFWCLenstronomy
import numpy as np
from copy import deepcopy

_tnfwc_lenstronomy = TNFWCLenstronomy()

class TNFWCHalo(Halo):
    """
    The base class for a cored and truncated NFW halo
    """
    _pseudo_nfw = True
    def __init__(self, mass, x, y, r3d, z,
                 sub_flag, lens_cosmo_instance, args,
                 truncation_class, concentration_class, unique_tag):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        self._lens_cosmo = lens_cosmo_instance
        self._concentration_class = concentration_class
        self._truncation_class = truncation_class
        self.tnfwc_lenstronomy = TNFWCLenstronomy()
        mdef = 'TNFWC'
        super(TNFWCHalo, self).__init__(mass, x, y, r3d, mdef, z, sub_flag,
                                           lens_cosmo_instance, args, unique_tag)

    def density_profile_3d(self, r, profile_args=None):
        """

        :param r:
        :param profile_args:
        :return:
        """
        return self.density_profile_3d_lenstronomy(r)

    def density_profile_3d_lenstronomy(self, r, kwargs_lenstronomy=None):
        """
        Computes the 3-D density profile of the halo
        :param r: distance from center of halo [kpc]
        :return: the density profile in units M_sun / kpc^3
        """
        if kwargs_lenstronomy is None:
            kwargs_lenstronomy = self.lenstronomy_params[0][0]
        kpc_per_arcsec = self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)
        sigma_crit_mpc = self._lens_cosmo.get_sigma_crit_lensing(self.z, self._lens_cosmo.z_source)
        sigma_crit_kpc = sigma_crit_mpc * 1e-6
        factor = sigma_crit_kpc / kpc_per_arcsec
        return factor * self.tnfwc_lenstronomy.density_lens(r / kpc_per_arcsec,
                                                            kwargs_lenstronomy['Rs'],
                                                            kwargs_lenstronomy['alpha_Rs'],
                                                            kwargs_lenstronomy['r_core'],
                                                            kwargs_lenstronomy['r_trunc'])

    @property
    def lenstronomy_ID(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """

        return ['TNFWC']

    @property
    def c(self):
        """
        Computes the halo concentration (once)
        """
        if not hasattr(self, '_c'):
            self._c = self._concentration_class.nfw_concentration(self.mass, self.z_eval)
        return self._c

    @property
    def halo_effective_age(self):
        if not hasattr(self, '_halo_effective_age'):
            if self.is_subhalo:
                self._halo_effective_age = self.lens_cosmo.sidm_halo_effective_age(self.z,
                                                                               self.z_infall,
                                                                               self._args['lambda_t'])
            else:
                self._halo_effective_age = self.halo_age
        return self._halo_effective_age

    @property
    def sidm_timescale(self):
        """
        Computes the timescale given by Equation 2.2 in https://arxiv.org/pdf/2305.16176.pdf
        :return:
        """
        return self._args['sidm_timescale']

    @property
    def t_over_tc(self):
        """
        Computes the dimensionless timescale for the halo evolution
        :return:
        """
        return self.halo_effective_age / self.sidm_timescale

    @property
    def lenstronomy_params(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_kwargs_lenstronomy'):

            [alpha_Rs, rs_kpc, rc_kpc, rt_kpc, _] = self.profile_args
            dd = self.lens_cosmo.cosmo.D_A_z(self.z)
            Rs_angle = rs_kpc * 1e-3 / dd / self.lens_cosmo._arcsec  # Rs in arcsec
            r_core_angle = rc_kpc * 1e-3 / dd / self.lens_cosmo._arcsec
            r_trunc_angle = rt_kpc * 1e-3 / dd / self.lens_cosmo._arcsec
            x, y = np.round(self.x, 4), np.round(self.y, 4)
            kwargs = [{'alpha_Rs': alpha_Rs,
                       'Rs': Rs_angle,
                       'center_x': x, 'center_y': y,
                       'r_trunc': r_trunc_angle,
                       'r_core': r_core_angle}]
            self._kwargs_lenstronomy = kwargs
        return self._kwargs_lenstronomy, None

    @property
    def profile_args(self):
        """
        Computes the time-evolving profile parameters
        :return: the density normalization, scale radius, and core radius for thee SIDM halo
        """
        if not hasattr(self, '_profile_args'):
            rt_kpc = self._truncation_class.truncation_radius_halo(self)
            _, rs_0, _ = self.lens_cosmo.NFW_params_physical(self.mass, self.c, self.z_eval,
                                                             pseudo_nfw=self._pseudo_nfw)
            rs_kpc, rc_kpc = evolve_profile(self.t_over_tc, rs_0)
            kpc_per_arcsec = self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)
            r200_kpc = self.c * rs_0
            # now solve for rho_s to conserve total mass
            Rs_angle = rs_kpc / kpc_per_arcsec  # Rs in arcsec
            r_core_angle = rc_kpc / kpc_per_arcsec
            r_trunc_angle = rt_kpc / kpc_per_arcsec
            x = np.logspace(-4,
                            np.log10(self.c),
                            1000)
            r = x * rs_0
            kwargs_temp = {'alpha_Rs': 1.0,
                           'Rs': Rs_angle,
                           'r_core': r_core_angle,
                           'r_trunc': r_trunc_angle}
            rho = self.density_profile_3d_lenstronomy(r, kwargs_temp)
            mass_3d = np.trapz(4 * np.pi * r ** 2 * rho, r)
            alpha_Rs = self._args['mass_conservation'] / mass_3d
            self._profile_args = (alpha_Rs, rs_kpc, rc_kpc, rt_kpc, r200_kpc)
        return self._profile_args

    @property
    def params_physical(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_params_physical'):
            [rho_s, rs_kpc, rc_kpc, rt_kpc, r200_kpc] = self.profile_args
            self._params_physical = {'rhos': rho_s * self._rescale_norm,
                                     'rs': rs_kpc, 'r200': r200_kpc,
                                     'r_trunc_kpc': rt_kpc,
                                     'r_core_kpc': rc_kpc}
        return self._params_physical

    @property
    def vmax_nfw(self):
        """
        Returns the maximum circular velocity in km/sec of an NFW profile with given rhos, rs
        :return:
        """
        if not hasattr(self, '_vmax'):
            rhos, rs, _ = self.nfw_params
            _ = self.profile_args
            self._vmax = self._lens_cosmo.nfw_vmax(self._rescale_norm * rhos, rs)
        return self._vmax

class Hybrid(Halo):

    """
    A hybrid TNFW + TNFWC profile with a relative weighting between them
    """
    def __init__(self, tnfw_halo, tnfwc_halo, rescaling_factor):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        c = tnfw_halo.c
        tnfwc_halo._c = c
        if tnfw_halo.is_subhalo:
            assert tnfwc_halo.is_subhalo
            tnfwc_halo._z_infall = tnfw_halo.z_infall
        self.tnfw_halo = tnfw_halo
        self.tnfwc_halo = tnfwc_halo
        if hasattr(self.tnfw_halo, '_kwargs_lenstronomy'):
            delattr(self.tnfw_halo, '_kwargs_lenstronomy')
        if hasattr(self.tnfwc_halo, '_kwargs_lenstronomy'):
            delattr(self.tnfwc_halo, '_kwargs_lenstronomy')
        mdef = 'TNFWC_HYBRID'
        self._rescaling_factor = rescaling_factor
        super(Hybrid, self).__init__(self.tnfw_halo.mass, self.tnfw_halo.x, self.tnfw_halo.y,
                                     self.tnfw_halo.r3d, mdef, self.tnfw_halo.z,
                                     self.tnfw_halo.is_subhalo, self.tnfw_halo.lens_cosmo,
                                     self.tnfw_halo._args, self.tnfw_halo.unique_tag)

    @property
    def vmax_nfw(self):
        """
        Returns the maximum circular velocity in km/sec of an NFW profile with given rhos, rs
        :return:
        """
        return self.tnfw_halo.vmax_nfw

    @property
    def halo_effective_age(self):
        return self.tnfwc_halo.halo_effective_age

    @property
    def t_over_tc(self):
        """

        :return:
        """
        return self.tnfwc_halo.t_over_tc

    @property
    def sidm_timescale(self):
        """

        :return:
        """
        return self.tnfwc_halo.sidm_timescale

    @property
    def c(self):
        """
        Computes the halo concentration (once)
        """
        if not hasattr(self, '_c'):
            self._c = self.tnfw_halo.c
        return self._c

    def profile_args(self):
        return [self.tnfw_halo.profile_args(), self.tnfwc_halo.profile_args()]

    @property
    def lenstronomy_params(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_kwargs_lenstronomy'):
            lenstronomy_params_tnfw = deepcopy(self.tnfw_halo.lenstronomy_params[0])
            lenstronomy_params_tnfwc = deepcopy(self.tnfwc_halo.lenstronomy_params[0])
            lenstronomy_params_tnfw[0]['alpha_Rs'] *= 1-self._rescaling_factor
            lenstronomy_params_tnfwc[0]['alpha_Rs'] *= self._rescaling_factor
            self._kwargs_lenstronomy = lenstronomy_params_tnfw + lenstronomy_params_tnfwc
        return self._kwargs_lenstronomy, None

    @property
    def nfw_params(self):
        """
        Computes the nfw profile parameters (rs,rho_s) from mass and concentration
        :return: rs, r200 and rho_s in units kpc, kpc, and M_sun / kpc^3
        """
        if not hasattr(self, '_nfw_params'):
            self._nfw_params_tnfw = self.tnfw_halo.nfw_params
            self._nfw_params_tnfwc = self.tnfwc_halo.nfw_params
            self._nfw_params = [self._nfw_params_tnfw, self._nfw_params_tnfwc]
        return [self._nfw_params[0][0], self._nfw_params[1][0]], \
               [self._nfw_params[0][1], self._nfw_params[1][1]], \
               [self._nfw_params[0][2], self._nfw_params[1][2]]

    @property
    def lenstronomy_ID(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        return ['TNFW', 'TNFWC']

    def density_profile_3d_lenstronomy(self, r, profile_args=None):
        """
        Computes the 3-D density profile of the halo
        :param r: distance from center of halo [kpc]
        :return: the density profile in units M_sun / kpc^3
        """
        rho_tnfw = self.tnfw_halo.density_profile_3d_lenstronomy(r)
        rho_tnfwc = self.tnfwc_halo.density_profile_3d_lenstronomy(r)
        return rho_tnfw * (1-self._rescaling_factor) + rho_tnfwc * self._rescaling_factor

    def density_profile_3d(self, r, profile_args=None):
        """
        Computes the 3-D density profile of the halo
        :param r: distance from center of halo [kpc]
        :return: the density profile in units M_sun / kpc^3
        """
        rho_tnfw = self.tnfw_halo.density_profile_3d(r)
        rho_tnfwc = self.tnfwc_halo.density_profile_3d(r)
        return rho_tnfw * (1-self._rescaling_factor) + rho_tnfwc * self._rescaling_factor

def evolve_profile(t, rs_0):

    t = min(1.6, t)
    rs = rs_0 * rs_evolution(t)
    rc = rs_0 * rc_evolution(t)
    rc = max(1e-7 * rs, rc)
    return rs, rc

def rs_evolution(tr):
    if tr <= 1.0:
        return _rs_evolution(tr)
    else:
        return rs_extrapolation(tr)


def rc_evolution(tr):
    if tr <= 1.0:
        return _rc_evolution(tr)
    else:
        return rc_extrapolation(tr)


def rho_s_evolution(tr):
    """
    Computes the evolution of the density normalization for the profile
    :param t_over_tc: the physical time scaled by the characteristic evolution time
    :return: the density in units of rho_s_nfw
    """
    return 0.692896 + 1.11215 * tr + 4.08729 * tr ** 5 - 5.78038 * tr ** 7 + 5.30498 * tr ** 9 + (
        1 - 0.692896) / np.log(0.001) * np.log(0.001 + tr)


def _rs_evolution(tr):
    """
    Computes the evolution of the scale radius for the profile
    :param t_over_tc: the physical time scaled by the characteristic evolution time
    :return: the scale radius in units of rs_nfw
    """

    return 1.13293 - 0.518186 * tr + 0.30265 * tr ** 2 - 0.400463 * tr ** 3 + (1 - 1.13293) * np.log(
        0.001 + tr) / np.log(0.001)


def _rc_evolution(tr):
    """
    Computes the evolution of the core radius for the profile
    :param t_over_tc: the physical time scaled by the characteristic evolution time
    :return: the core radius in units of rs_nfw
    """

    return 1.06419 * np.sqrt(tr) - 1.33207 * tr + 0.431133 * tr ** 2 - 0.148087 * tr ** 3 + 0.024455 * tr ** 4


def rs_extrapolation(tr, t0=0.5, c1=0.47, c2=2.4):
    """

    :param tr:
    :return:
    """
    return rs_evolution(t0) * np.exp(-c1 * (tr - t0) - c2 * (tr - t0) ** 3)


def rc_extrapolation(t_over_tc, t0=0.5, c1=0.28):
    """

    :param t_over_tc:
    :param t0:
    :param c1:
    :return:
    """
    rc = rc_evolution(t0) - c1 * (t_over_tc - t0)
    if rc < 0.0001:
        rc = 0.0001
    return rc
