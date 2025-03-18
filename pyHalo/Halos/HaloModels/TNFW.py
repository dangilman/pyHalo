from pyHalo.Halos.halo_base import Halo
import numpy as np
from pyHalo.Halos.tnfw_halo_util import tnfw_mass_fraction
from lenstronomy.LensModel.Profiles.tnfw import TNFW

class TNFWFieldHalo(Halo):
    # we use the pseudo nfw methods to normalize profile
    _pseudo_nfw = False
    """
    The base class for a truncated NFW halo
    """
    def __init__(self, mass, x, y, r3d, z,
                 sub_flag, lens_cosmo_instance, args,
                 truncation_class, concentration_class, unique_tag):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        self._lens_cosmo = lens_cosmo_instance
        self._concentration_class = concentration_class
        self._truncation_class = truncation_class
        self.tnfw_lenstronomy = TNFW()
        mdef = 'TNFW'
        super(TNFWFieldHalo, self).__init__(mass, x, y, r3d, mdef, z, sub_flag,
                                           lens_cosmo_instance, args, unique_tag)

    @classmethod
    def simple_setup(cls, mass, x, y, z, tau, lens_cosmo,
                     concentration_model='DIEMERJOYCE19'):
        """
        Creates an instance of the TNFWFieldHalo class with some default assumptions for the concentration-mass relation
        and truncation
        :param mass: halo mass
        :param x: halo x-coordinate [arcsec]
        :param y: halo y-coordinate [arcsec]
        :param z: halo redshift
        :param tau: the truncation radius in units of rs
        :param lens_cosmo: an instance of lens_cosmo class
        :param concentration_model: the concentration-mass relation model
        :return: an instance of TNFWFieldHalo class
        """
        r3d = None
        sub_flag = False
        args = {}
        from pyHalo.concentration_models import preset_concentration_models
        from pyHalo.truncation_models import truncation_models
        _c_model, _ = preset_concentration_models(concentration_model)
        _t_model, _ = truncation_models('MULTIPLE_RS')
        concentration_class = _c_model(lens_cosmo.cosmo.astropy, scatter=False)
        truncation_class = _t_model(lens_cosmo, tau)
        return TNFWFieldHalo(mass, x, y, r3d, z, sub_flag, lens_cosmo, args,
                             truncation_class, concentration_class, 1.0)

    def density_profile_3d(self, r, profile_args=None, scaling=1.0):
        """
        Computes the 3-D density profile of the halo
        :param r: distance from center of halo [kpc]
        :return: the density profile in units M_sun / kpc^3
        """
        if profile_args is None:
            c, rt = self.profile_args
        else:
            c, rt = profile_args
        rhos, rs, _ = self.lens_cosmo.NFW_params_physical(self.mass, self.c, self.z_eval)
        tau = rt / rs
        x = r / rs
        rho_nfw = rhos / x / (1 + x) ** 2
        return scaling * rho_nfw * tau ** 2 / (tau ** 2 + x ** 2)

    def mass_3d(self, rmax, profile_args=None):
        """
        Calculate the enclosed mass in 3D
        :param rmax:
        :param profile_args:
        :return:
        """
        if rmax == 'r200':
            rmax = self.nfw_params[1] * self.c
        rs = self.nfw_params[1]
        rho0 = self.nfw_params[0] * self._rescale_norm
        r_trunc = self.profile_args[1]
        return self.tnfw_lenstronomy.mass_3d(rmax, rs, rho0, r_trunc/rs)

    def density_profile_3d_lenstronomy(self, r, profile_args=None):
        """
        Computes the 3-D density profile of the halo
        :param r: distance from center of halo [kpc]
        :return: the density profile in units M_sun / kpc^3
        """
        prof = self.tnfw_lenstronomy
        kwargs_lenstronomy = self.lenstronomy_params[0][0]
        kpc_per_arcsec = self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)
        sigma_crit_mpc = self._lens_cosmo.get_sigma_crit_lensing(self.z, self._lens_cosmo.z_source)
        sigma_crit_kpc = sigma_crit_mpc * 1e-6
        factor = sigma_crit_kpc / kpc_per_arcsec
        rhos = prof.alpha2rho0(kwargs_lenstronomy['alpha_Rs'], kwargs_lenstronomy['Rs'])
        return factor*prof.density(r / kpc_per_arcsec, kwargs_lenstronomy['Rs'],
                                                    rhos,
                                                    kwargs_lenstronomy['r_trunc'])

    @property
    def lenstronomy_ID(self):
            """
            See documentation in base class (Halos/halo_base.py)
            """

            return ['TNFW']

    @property
    def c(self):
        """
        Computes the halo concentration (once)
        """

        if not hasattr(self, '_c'):
            self._c = self._concentration_class.nfw_concentration(self.mass, self.z_eval)
        return self._c

    @property
    def params_physical(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """

        if not hasattr(self, '_params_physical'):
            [_, rt] = self.profile_args
            rhos, rs, r200 = self.nfw_params
            self._params_physical = {'rhos': rhos * self._rescale_norm, 'rs': rs, 'r200': r200, 'r_trunc_kpc': rt}

        return self._params_physical

    @property
    def lenstronomy_params(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_kwargs_lenstronomy'):

            [_, rt] = self.profile_args
            rhos_kpc, rs_kpc, _ = self.nfw_params
            rhos_mpc = rhos_kpc * 1e9
            rs_mpc = rs_kpc * 1e-3
            Rs_angle, theta_Rs = self._lens_cosmo.nfw_physical2angle_fromNFWparams(rhos_mpc,
                                                                                   rs_mpc,
                                                                                   self.z)
            x, y = np.round(self.x, 4), np.round(self.y, 4)
            Rs_angle = np.round(Rs_angle, 10)
            theta_Rs = np.round(theta_Rs, 10)
            r_trunc_arcsec = rt / self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)
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
            truncation_radius_kpc = self._truncation_class.truncation_radius_halo(self)
            self._profile_args = (self.c, truncation_radius_kpc)
        return self._profile_args

    @property
    def vmax_nfw(self):
        """
        Returns the maximum circular velocity in km/sec
        :return:
        """
        if not hasattr(self, '_vmax'):
            rhos, rs, _ = self.nfw_params
            _ = self.profile_args
            self._vmax = self._lens_cosmo.nfw_vmax(self._rescale_norm * rhos, rs)
        return self._vmax

    @property
    def pseudo_nfw(self):
        return False

class TNFWSubhalo(TNFWFieldHalo):
    """
    Defines a truncated NFW halo that is a subhalo of the host dark matter halo
    """

    @property
    def bound_mass(self):
        """
        Computes the mass inside the infall virial radius (with truncation effects included)
        :return: the mass inside r = c * r_s
        """
        if hasattr(self, '_kwargs_lenstronomy'):
            tau = self._kwargs_lenstronomy[0]['r_trunc'] / self._kwargs_lenstronomy[0]['Rs']
        else:
            params_physical = self.params_physical
            tau = params_physical['r_trunc_kpc'] / params_physical['rs']
        f = tnfw_mass_fraction(tau, self.c)
        return f * self.mass * self._rescale_norm

    @property
    def bound_mass_galacticus_definition(self):
        """
        Computes the mass inside the virial radius (with truncation effects included)
        :return: the mass inside r = c * r_s
        """
        if self._truncation_class.name in ['TruncationGalacticus', 'TruncationGalacticusKeeley24']:
            pass
        else:
            raise Exception('this method can only be called when using the TruncationGalacticus class')

        if not hasattr(self, '_mbound_galacticus_definition'):
            self._mbound_galacticus_definition = self._truncation_class.calculate_mbound(self)
        return self._mbound_galacticus_definition

    @property
    def profile_args(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_profile_args'):
            if self._truncation_class.name == 'TruncationGalacticus':
                mbound = self.bound_mass_galacticus_definition
                truncation_radius_kpc = self._truncation_class.truncation_radius_from_bound_mass(self, mbound)
            else:
                truncation_radius_kpc = self._truncation_class.truncation_radius_halo(self)
            self._profile_args = (self.c, truncation_radius_kpc)
        return self._profile_args

    @property
    def pseudo_nfw(self):
        return False
