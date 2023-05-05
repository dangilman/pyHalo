from pyHalo.Halos.halo_base import Halo
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator


def tnfw_mass_fraction(tau, c):
    """
    This function returns the fraction = final_mass/initial_mass, assuming a truncated NFW profile
    :param tau: the truncation radius in units of the scale radius
    :param c: the halo concentration
    """
    x = c
    Rs = 1.0
    r_trunc = tau * Rs
    func = (r_trunc ** 2 * (-2 * x * (1 + r_trunc ** 2) + 4 * (1 + x) * r_trunc * np.arctan(x / r_trunc) -
                            2 * (1 + x) * (-1 + r_trunc ** 2) * np.log(Rs) + 2 * (1 + x) * (-1 + r_trunc ** 2) * np.log(
            Rs * (1 + x)) +
                            2 * (1 + x) * (-1 + r_trunc ** 2) * np.log(Rs * r_trunc) -
                            (1 + x) * (-1 + r_trunc ** 2) * np.log(Rs ** 2 * (x ** 2 + r_trunc ** 2)))) / (
                   2. * (1 + x) * (1 + r_trunc ** 2) ** 2)
    mass_loss = func / (np.log(1 + c) - c / (1 + c))
    return mass_loss


def tau_mf_interpolator():
    N = 250
    tau = np.logspace(-3.5, 2.5, N)
    concentration = np.linspace(1.0, 200.0, N)

    log10_mass_fraction_1d = np.linspace(-4, -0.001, N)
    log10tau_2d = np.zeros((N, N))

    # This computes the value of tau that corresponds to each pair of (concentration, mass_loss)
    for i, con_i in enumerate(concentration):
        mfinal = tnfw_mass_fraction(tau, con_i)
        log10final_mass = np.log10(mfinal)
        mfinterp = interp1d(log10final_mass, np.log10(tau), fill_value='extrapolate')

        for j, log10_mass_j in enumerate(log10_mass_fraction_1d):
            log10tau_2d[i, j] = float(mfinterp(log10_mass_j))

    interp_points = (concentration, log10_mass_fraction_1d)
    interpolator = RegularGridInterpolator(interp_points, log10tau_2d, fill_value=None, bounds_error=False)
    return interpolator

_truncation_radius_interpolator = tau_mf_interpolator()

class TNFWSubhaloEmulator(Halo):
    """
    Defines a truncated NFW halo that is a subhalo of the host dark matter halo
    """

    def __init__(self, infall_mass, x, y, final_bound_mass, infall_concentration, redshift,
                 lens_cosmo_instance, unique_tag=None):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        self._lens_cosmo = lens_cosmo_instance

        r3d = None
        profile_definition = 'TNFW'
        sub_flag = True
        args = None
        if unique_tag is None:
            unique_tag = np.random.rand()

        # set the concentration
        self.c = infall_concentration
        self._bound_mass_fraction = final_bound_mass / infall_mass
        self._kpc_per_arcsec_at_z = self._lens_cosmo.cosmo.kpc_proper_per_asec(redshift)
        x_arcsec = x / self._kpc_per_arcsec_at_z
        y_arcsec = y / self._kpc_per_arcsec_at_z
        super(TNFWSubhaloEmulator, self).__init__(infall_mass, x_arcsec, y_arcsec, r3d, profile_definition, redshift,
                                                  sub_flag,
                                                  lens_cosmo_instance, args, unique_tag)

    @property
    def z_eval(self):
        """
        Returns the halo redshift
        """

        return self.z

    @property
    def lenstronomy_params(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_kwargs_lenstronomy'):
            [concentration, rt] = self.profile_args
            # evaluate density parameters at the time of lensing
            _rhos_mpc, _rs_mpc, _ = self._lens_cosmo.nfwParam_physical(self.mass, concentration, self.z_eval)
            # convert to angles at the time of lensing (deflector redshift)
            Rs_angle, theta_Rs = self._lens_cosmo.nfw_physical2angle(self.mass, concentration, self.z_eval)
            x, y = np.round(self.x, 4), np.round(self.y, 4)
            Rs_angle = np.round(Rs_angle, 10)
            theta_Rs = np.round(theta_Rs, 10)
            r_trunc_arcsec = rt / self._kpc_per_arcsec_at_z
            kwargs = [{'alpha_Rs': self._rescale_norm * theta_Rs, 'Rs': Rs_angle,
                       'center_x': x, 'center_y': y, 'r_trunc': r_trunc_arcsec}]
            self._kwargs_lenstronomy = kwargs

        return self._kwargs_lenstronomy, None

    @property
    def profile_args(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_profile_args'):
            point = (self.c, np.log10(self._bound_mass_fraction))
            Rs_angle, _ = self._lens_cosmo.nfw_physical2angle(self.mass, self.c, self.z)
            log10_tau = float(_truncation_radius_interpolator(point))
            rt_over_rs = 10 ** log10_tau
            truncation_kpc = Rs_angle * rt_over_rs * self._kpc_per_arcsec_at_z
            self._profile_args = (self.c, truncation_kpc)

        return self._profile_args

    @property
    def params_physical(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_params_physical'):
            [concentration, rt] = self.profile_args
            rhos, rs, r200 = self._lens_cosmo.NFW_params_physical(self.mass, concentration, self.z)
            self._params_physical = {'rhos': rhos, 'rs': rs, 'r200': r200, 'r_trunc_kpc': rt}

        return self._params_physical

    @property
    def lenstronomy_ID(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """

        return ['TNFW']
