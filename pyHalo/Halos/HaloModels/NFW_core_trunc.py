from pyHalo.Halos.halo_base import Halo
from lenstronomy.LensModel.Profiles.nfw_core_truncated import TNFWC as TNFWLenstronomy
import numpy as np


class _TNFWCBaseClass(Halo):
    # we use the pseudo nfw methods to normalize profile
    _pseudo_nfw = True
    _rt_kpc = None
    """
    The base class for a cored and truncated NFW halo
    """
    _scale_evolution_timescale = 1.0
    # the parametric model by Yang et al. (2023) doesn't conserve mass; if this is set to True,
    # then the density profile of each lens is rescaled such that the mass inside the CDM halo equivalent virial radius
    # is conserved.
    def __init__(self, mass, x, y, r3d, z,
                 sub_flag, lens_cosmo_instance, args,
                 truncation_class, concentration_class, unique_tag):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        self._lens_cosmo = lens_cosmo_instance
        self._concentration_class = concentration_class
        self._truncation_class = truncation_class
        self._tnfwc_lenstronomy = TNFWLenstronomy()
        mdef = 'TNFWC'
        super(_TNFWCBaseClass, self).__init__(mass, x, y, r3d, mdef, z, sub_flag,
                                           lens_cosmo_instance, args, unique_tag)

    def density_profile_3d(self, r, profile_args=None):
        """
        Computes the 3-D density profile of the halo
        :param r: distance from center of halo [kpc]
        :return: the density profile in units M_sun / kpc^3
        """
        if profile_args is None:
            [rho_s_kpc, rs_kpc, rc_kpc, rt_kpc, _] = self.profile_args()
        else:
            [rho_s_kpc, rs_kpc, rc_kpc, rt_kpc, _] = profile_args
        tau = rt_kpc / rs_kpc
        beta = rc_kpc / rs_kpc
        x = r / rs_kpc
        return self._rescale_norm * rho_s_kpc * tau ** 2 / (beta ** 2 + x**2) ** 0.5 / (tau ** 2 + x ** 2) / (1 + x ** 2)

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

class TNFWCFieldHaloSIDM(_TNFWCBaseClass):

    """
    Implements a temporal evolution of the halo profile based on Yang et al. (2023)
    https://arxiv.org/pdf/2305.16176.pdf
    """
    _regularize_t_over_tc = 1.0 # don't evolve profile past this time because fitting function breaks down
    def __init__(self, mass, x, y, r3d, z,
                 sub_flag, lens_cosmo_instance, args,
                 truncation_class, concentration_class, unique_tag):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        super(TNFWCFieldHaloSIDM, self).__init__(mass, x, y, r3d, z,
                 sub_flag, lens_cosmo_instance, args,
                 truncation_class, concentration_class, unique_tag)

    @property
    def halo_effective_age(self):
        return self.halo_age

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

            arcsec = 2 * np.pi / 360 / 3600
            [rhos_kpc, rs_kpc, rc_kpc, rt_kpc, _] = self.profile_args()
            dd_kpc = self.lens_cosmo.cosmo.D_A_z(self.z) * 1e3
            Rs_angle, theta_Rs = self._lens_cosmo.nfw_physical2angle_fromNFWparams(1e9*rhos_kpc,
                                                                                   1e-3*rs_kpc,
                                                                                   self.z,
                                                                                   pseudo_nfw=self._pseudo_nfw)
            r_trunc_angle = rt_kpc / dd_kpc / arcsec
            r_core_angle = max(1e-6 * Rs_angle, rc_kpc / dd_kpc / arcsec)
            x, y = np.round(self.x, 4), np.round(self.y, 4)
            kwargs = [{'alpha_Rs': self._rescale_norm * theta_Rs, 'Rs': Rs_angle,
                       'center_x': x, 'center_y': y, 'r_trunc': r_trunc_angle, 'r_core': r_core_angle}]
            self._kwargs_lenstronomy = kwargs

        return self._kwargs_lenstronomy, None

    def set_tidal_evolution(self, rt_kpc, rescale_norm):
        """
        Allows one to manually set the tidal evolution to circument a second evaluation of truncation model
        :param rt_kpc: truncation radius [kpc]
        :param rescale_norm: rescales the overall normalization
        :return:
        """
        self._rt_kpc = rt_kpc
        self._rescale_norm = rescale_norm
        self._rescaled_once = True
        if hasattr(self, '_kwargs_lenstronomy'):
            delattr(self, '_kwargs_lenstronomy')

    def profile_args(self):
        """
        I temporarily fix this by rescaling the normalization to conserve mass at all times

        Computes the time-evolving profile parameters
        :return: the density normalization, scale radius, and core radius for thee SIDM halo
        """
        if not hasattr(self, '_profile_args'):
            t = self.t_over_tc
            if self._rt_kpc is None:
                self._rt_kpc = self._truncation_class.truncation_radius_halo(self)
            rt_kpc = self._rt_kpc
            rho_s, rs_kpc, rc_kpc = self.get_params(t)
            r200_0 = self.nfw_params[-1]
            self._profile_args = (rho_s, rs_kpc, rc_kpc, rt_kpc, r200_0)
        return self._profile_args

    def get_params(self, t_over_tc):
        """

        :param t:
        :return:
        """
        rhos_0, rs_0, r200_0 = self.nfw_params
        t_cut = 1.001
        t_max = 1.7
        t_over_tc = min(t_max, t_over_tc)

        if t_over_tc < 0.1:
            rho_s = rhos_0
            rs_kpc = rs_0
            rc_kpc = 0.0
        elif t_over_tc < t_cut:
            rho_s = rhos_0 * rho_s_evolution(t_over_tc)
            rs_kpc = rs_0 * rs_evolution(t_over_tc)
            rc_kpc = rs_0 * rc_evolution(t_over_tc)
        else:
            rhos_0, rs_0, r200_0 = self.nfw_params
            rhos_last = rhos_0 * rho_s_evolution(t_cut)
            rs_kpc_last = rs_0 * rs_evolution(t_cut)
            rc_kpc_last = rs_0 * rc_evolution(t_cut)
            profile_args = (rhos_last, rs_kpc_last, rc_kpc_last, 100 * r200_0, r200_0)
            total_mass = self.mass_3d(r200_0, profile_args)
            rc_kpc = rs_0 * rc_extrapolation(t_over_tc)
            rs_kpc = rs_0 * rs_extrapolation(t_over_tc)
            rho_s = self.rhos_extrapolation(total_mass, rs_kpc, rc_kpc)

        return rho_s, rs_kpc, rc_kpc

    def rhos_extrapolation(self, total_mass, rs_kpc, rc_kpc):
        """

        :return:
        """
        r200_0 = self.nfw_params[-1]
        profile_args_integral = (1.0, rs_kpc, rc_kpc, 100 * r200_0, r200_0)
        m_integral = self.mass_3d(r200_0, profile_args_integral)
        return total_mass / m_integral

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
                                     'r_core_kcp': rc_kpc}
        return self._params_physical

class TNFWCSubhaloSIDM(TNFWCFieldHaloSIDM):

    """
    Implements a temporal evolution of the halo profile based on Yang et al. (2023)
    https://arxiv.org/pdf/2305.16176.pdf
    """
    def __init__(self, mass, x, y, r3d, z,
                 sub_flag, lens_cosmo_instance, args,
                 truncation_class, concentration_class, unique_tag):

        if 'scale_evolution_timescale' in args.keys():
            self._scale_evolution_timescale = args['scale_evolution_timescale']
        else:
            self._scale_evolution_timescale = 1.0

        super(TNFWCSubhaloSIDM, self).__init__(mass, x, y, r3d, z,
                 sub_flag, lens_cosmo_instance, args,
                 truncation_class, concentration_class, unique_tag)

    @property
    def halo_effective_age(self):
        if not hasattr(self, '_halo_effective_age'):
            self._halo_effective_age = self.lens_cosmo.sidm_halo_effective_age(self.z,
                                                                               self.z_infall,
                                                                               self._args['lambda_t'])
        return self._halo_effective_age

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
        self.tnfw_halo = tnfw_halo
        self.tnfwc_halo = tnfwc_halo
        mdef = 'TNFWC_HYBRID'
        self._rescaling_factor = rescaling_factor
        super(Hybrid, self).__init__(self.tnfw_halo.mass, self.tnfw_halo.x, self.tnfw_halo.y,
                                     self.tnfw_halo.r3d, mdef, self.tnfw_halo.z,
                                     self.tnfw_halo.is_subhalo, self.tnfw_halo.lens_cosmo,
                                     self.tnfw_halo._args, self.tnfw_halo.unique_tag)

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
            lenstronomy_params_tnfw = self.tnfw_halo.lenstronomy_params[0]
            lenstronomy_params_tnfwc = self.tnfwc_halo.lenstronomy_params[0]
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

    def density_profile_3d(self, r, profile_args=None):
        """
        Computes the 3-D density profile of the halo
        :param r: distance from center of halo [kpc]
        :return: the density profile in units M_sun / kpc^3
        """
        rho_tnfw = self.tnfw_halo.density_profile_3d(r)
        rho_tnfwc = self.tnfwc_halo.density_profile_3d(r)
        return rho_tnfw * (1-self._rescaling_factor) + rho_tnfwc * self._rescaling_factor

class HybridSubhalo(Hybrid):

    def __init__(self, tnfw_halo, tnfwc_halo, rescaling_factor):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        tnfwc_halo._z_infall = tnfw_halo.z_infall
        super(HybridSubhalo, self).__init__(tnfw_halo, tnfwc_halo, rescaling_factor)

def _check_valid_cross_section(cross_section):
    """

    :param cross_section:
    :return:
    """
    if getattr(cross_section, 'effective_cross_section') is None:
        raise Exception(
            'the specified SIDM_CROSS_SECTION class must have a method called effective_cross_section that takes'
            'as input a velocity and returns a cross section amplitude in cm^2 / gram')
    if not callable(cross_section.effective_cross_section):
        raise Exception('the effective_cross_section method in the cross section class must be a callable function that '
                        'takes an input a velocity and returns a cross section amplitude in cm^2 / gram.')

def rho_s_evolution(tr):
    """
    Computes the evolution of the density normalization for the profile
    :param t_over_tc: the physical time scaled by the characteristic evolution time
    :return: the density in units of rho_s_nfw
    """

    return 0.692896 + 1.11215 * tr + 4.08729 * tr ** 5 - 5.78038 * tr ** 7 + 5.30498 * tr ** 9 + (
        1 - 0.692896) / np.log(0.001) * np.log(0.001 + tr)

def rs_evolution(tr):
    """
    Computes the evolution of the scale radius for the profile
    :param t_over_tc: the physical time scaled by the characteristic evolution time
    :return: the scale radius in units of rs_nfw
    """

    return 1.13293 - 0.518186 * tr + 0.30265 * tr ** 2 - 0.400463 * tr ** 3 + (1 - 1.13293) * np.log(
        0.001 + tr) / np.log(0.001)

def rc_evolution(tr):
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
    if rc < 0.00001:
        rc = 0.00001
    return rc
