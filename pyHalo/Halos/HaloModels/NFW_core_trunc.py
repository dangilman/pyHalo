from pyHalo.Halos.halo_base import Halo
from lenstronomy.LensModel.Profiles.nfw_core_truncated import TNFWC as TNFWLenstronomy
import numpy as np
from scipy.interpolate import interp1d

_t_over_tc = np.array([4.30746710e-05, 5.29934226e-05, 6.51961529e-05, 8.02087909e-05,
                      9.86783705e-05, 1.21400918e-04, 1.49355758e-04, 1.83747724e-04,
                      2.26059087e-04, 2.78113435e-04, 3.42154274e-04, 4.20941718e-04,
                      5.17871450e-04, 6.37121073e-04, 7.83830159e-04, 9.64321766e-04,
                      1.18637495e-03, 1.45956006e-03, 1.79565116e-03, 2.20913354e-03,
                      2.71782801e-03, 3.34365892e-03, 4.11359915e-03, 5.06083257e-03,
                      6.22618426e-03, 7.65988006e-03, 9.42371123e-03, 1.15936976e-02,
                      1.42633641e-02, 1.75477715e-02, 2.15884752e-02, 2.65596269e-02,
                      3.26754795e-02, 4.01996220e-02, 4.94563395e-02, 6.08445901e-02,
                      7.48551990e-02, 9.20920137e-02, 1.13297928e-01, 1.39386903e-01,
                      1.71483354e-01, 2.10970615e-01, 2.59550560e-01, 3.19316948e-01,
                      3.92845669e-01, 4.83305758e-01, 5.94595981e-01, 7.31512867e-01,
                      8.99957435e-01, 1.10718953e+00])
_rescale_factor = np.array([1., 0.99952292, 0.99894231, 0.99823652, 0.99737973,
                           0.99634139, 0.9950855, 0.9935701, 0.99174668, 0.98955987,
                           0.98694731, 0.98383994, 0.9801627, 0.97583573, 0.97077622,
                           0.96490061, 0.95812722, 0.95037906, 0.94158659, 0.93169024,
                           0.92064258, 0.90841018, 0.89497524, 0.88033706, 0.86451376,
                           0.84754437, 0.82949143, 0.81044447, 0.79052418, 0.76988764,
                           0.74873421, 0.72731234, 0.70592668, 0.68494528, 0.6648062,
                           0.64602217, 0.62918191, 0.61494518, 0.60402759, 0.59716819,
                           0.59506842, 0.59828068, 0.6070057, 0.62072157, 0.63753333,
                           0.65330003, 0.66209406, 0.66588949, 0.69974511, 0.64344236])
_rescale_density_interp = interp1d(_t_over_tc, _rescale_factor)

class _DensityRescaling(object):

    def __init__(self, interp_function):
        self._func = interp_function
    def __call__(self, tovertc):
        tmin, tmax = 4.30746710e-05, 1.10718953e+00
        rescale_max = 0.64344236
        if isinstance(tovertc, int) or isinstance(tovertc, float):
            if tovertc <= tmin:
                return 1.0
            elif tovertc >= tmax:
                return rescale_max
            else:
                return float(self._func(tovertc))
        else:
            rescale = np.ones_like(tovertc)
            inds_high = np.where(tovertc >= tmax)[0]
            inds = np.where(np.logical_and(tovertc > tmin, tovertc < tmax))[0]
            rescale[inds_high] = rescale_max
            rescale[inds] = self._func(tovertc)
            return np.array(rescale)
_density_scale = _DensityRescaling(_rescale_density_interp)

class _TNFWCBaseClass(Halo):

    """
    The base class for a cored and truncated NFW halo
    """
    enforce_mass_conservation = False
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
        return rho_s_kpc * tau ** 2 / (beta ** 2 + x**2) ** 0.5 / (tau ** 2 + x ** 2) / (1 + x ** 2)

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
        # This class must have a cross section model supplied through the kwargs_special argument
        self._sidm_cross_section = args['SIDM_CROSS_SECTION']
        _check_valid_cross_section(self._sidm_cross_section)
        super(TNFWCFieldHaloSIDM, self).__init__(mass, x, y, r3d, z,
                 sub_flag, lens_cosmo_instance, args,
                 truncation_class, concentration_class, unique_tag)

    @property
    def sidm_timescale(self):
        """
        Computes the timescale given by Equation 2.2 in https://arxiv.org/pdf/2305.16176.pdf
        :return:
        """
        if not hasattr(self, '_sidm_timescale'):
            rho_s, rs, _ = self.nfw_params
            C = 0.75
            G = 4.3e-6
            conversion_fac = 2.135788e-10
            vmax = 1.65 * np.sqrt(G * rho_s * rs ** 2) # fixed for an NFW profile
            v_scale = np.sqrt(4 * np.pi * G * rho_s * rs ** 2)
            sigma_eff = self._sidm_cross_section.effective_cross_section(0.64 * vmax)
            denom = sigma_eff * rho_s * v_scale * conversion_fac
            self._sidm_timescale = self._scale_evolution_timescale * 150 / C / denom
        return self._sidm_timescale

    @property
    def t_over_tc(self):
        """
        Computes the dimensionless timescale for the halo evolution
        :return:
        """
        return self.halo_age / self.sidm_timescale

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
                                                                                   pseudo_nfw=True)
            r_trunc_angle = rt_kpc / dd_kpc / arcsec
            r_core_angle = max(1e-6 * Rs_angle, rc_kpc / dd_kpc / arcsec)
            x, y = np.round(self.x, 4), np.round(self.y, 4)
            kwargs = [{'alpha_Rs': self._rescale_norm * theta_Rs, 'Rs': Rs_angle,
                       'center_x': x, 'center_y': y, 'r_trunc': r_trunc_angle, 'r_core': r_core_angle}]
            self._kwargs_lenstronomy = kwargs

        return self._kwargs_lenstronomy, None

    def profile_args(self):
        """
        TODO: the parameterization for the density profile presented by Yang et al. doesn't conserve mass.
        I temporarily fix this by rescaling the normalization to conserve mass at all times

        Computes the time-evolving profile parameters
        :return: the density normalization, scale radius, and core radius for thee SIDM halo
        """
        if not hasattr(self, '_profile_args'):
            t = min(self._regularize_t_over_tc, self.t_over_tc)
            rhos_0, rs_0, r200_0 = self.nfw_params
            rho_s = rhos_0 * rho_s_evolution(t)
            rs_kpc = rs_0 * rs_evolution(t)
            rc_kpc = rs_0 * rc_evolution(t)
            rt_kpc = self._truncation_class.truncation_radius_halo(self)
            #
            # if self.enforce_mass_conservation:
            #     profile_args = (rho_s, rs_kpc, rc_kpc, 1000 * rs_kpc, r200_0)
            #     rescale_density = self.mass / self.mass_3d(r200_0, profile_args)
            # else:
            #     rescale_density = 1.0
            # print(rescale_density)
            self._profile_args = (1.0 * rho_s, rs_kpc, rc_kpc, rt_kpc, r200_0)
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
                                     'r_core_kcp': rc_kpc}
        return self._params_physical

class TNFWCSubhaloSIDM(TNFWCFieldHaloSIDM):

    """
    Implements a temporal evolution of the halo profile based on Yang et al. (2023)
    https://arxiv.org/pdf/2305.16176.pdf

    TODO: add something here to account for tidal effects
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
    return 0.692896 + 1.11215 * tr + 4.08729 * tr ** 5 - 5.78038 * tr ** 7 + 5.30498 * tr ** 9 + (
            1 - 0.692896) / np.log(0.001) * np.log(0.001 + tr)


def rs_evolution(tr):
    return 1.13293 - 0.518186 * tr + 0.30265 * tr ** 2 - 0.400463 * tr ** 3 + (1 - 1.13293) * np.log(
        0.001 + tr) / np.log(0.001)


def rc_evolution(tr):
    # 1.13559411 -1.42145504  0.46006192 -0.15802389  0.02609586
    return 1.06419 * np.sqrt(tr) - 1.33207 * tr + 0.431133 * tr ** 2 - 0.148087 * tr ** 3 + 0.024455 * tr ** 4

#
# def rho_s_evolution(t_over_tc):
#     """
#     Computes the evolution of the density normalization for the profile
#     :param t_over_tc: the physical time scaled by the characteristic evolution time
#     :return: the density in units of rho_s_nfw
#     """
#
#     c1 = 0.101380
#     reg = (1 - c1) * np.log(t_over_tc + 0.001) / np.log(0.001)
#     return c1 + 1.2075 * t_over_tc + 0.5016 * t_over_tc ** 5 + 0.4897 * t_over_tc ** 7 + 1.8380 * t_over_tc ** 9 + reg
#
# def rs_evolution(t_over_tc):
#     """
#     Computes the evolution of the scale radius for the profile
#     :param t_over_tc: the physical time scaled by the characteristic evolution time
#     :return: the scale radius in units of rs_nfw
#     """
#
#     c1 = 1.273
#     reg = (1 - c1) * np.log(t_over_tc + 0.001) / np.log(0.001)
#     return c1 - 0.7551 * t_over_tc + 0.5775 * t_over_tc ** 2 - 0.5437 * t_over_tc**3 + reg
#
# def rc_evolution(t_over_tc):
#     """
#     Computes the evolution of the core radius for the profile
#     :param t_over_tc: the physical time scaled by the characteristic evolution time
#     :return: the core radius in units of rs_nfw
#     """
#
#     return 1.136 * np.sqrt(t_over_tc) - 1.421 * t_over_tc + 0.4601 * t_over_tc**2 - 0.1580*t_over_tc**3 +0.0261*t_over_tc**4
