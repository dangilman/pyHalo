from pyHalo.Halos.halo_base import Halo
from lenstronomy.LensModel.Profiles.splcore import SPLCORE
import numpy as np

class PowerLawSubhalo(Halo):

    """
    The base class for a halo modeled as a power law profile
    """
    def __init__(self, mass, x, y, r3d, z,
                 sub_flag, lens_cosmo_instance, args, truncation_class,
                 concentration_class, unique_tag):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        self._prof = SPLCORE()
        self._lens_cosmo = lens_cosmo_instance
        self._truncation_class = truncation_class
        self._concentration_class = concentration_class
        mdef = 'SPL_CORE'
        super(PowerLawSubhalo, self).__init__(mass, x, y, r3d, mdef, z, sub_flag,
                                              lens_cosmo_instance, args, unique_tag)

    @property
    def lenstronomy_params(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_lenstronomy_args'):

            (concentration, gamma, x_core_halo) = self.profile_args
            rhos, rs, r200 = self._lens_cosmo.NFW_params_physical(self.mass, concentration, self.z_eval)
            kpc_per_arcsec = self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)

            if 'x_match' in self._args.keys():
                if self._args['x_match'] == 'c':
                    x_match = concentration
                else:
                    x_match = self._args['x_match']
            else:
                # r_vmax = 2.16 * rs
                x_match = 2.16 # r_vmax
            r_match_arcsec = x_match * rs / kpc_per_arcsec
            fx = np.log(1+x_match) - x_match/(1 + x_match)
            m = 4 * np.pi * rs ** 3 * rhos * fx
            r_core_arcsec = x_core_halo * r_match_arcsec / x_match
            sigma_crit_mpc = self._lens_cosmo.get_sigma_crit_lensing(self.z, self._lens_cosmo.z_source)
            sigma_crit_arcsec = sigma_crit_mpc * (0.001 * kpc_per_arcsec) ** 2
            rho0 = m/self._prof.mass_3d(r_match_arcsec, sigma_crit_arcsec, r_core_arcsec, gamma)
            sigma0 = rho0 * r_core_arcsec
            self._lenstronomy_args = [{'sigma0': sigma0, 'gamma': gamma, 'center_x': self.x, 'center_y': self.y,
                                      'r_core': r_core_arcsec}]

        return self._lenstronomy_args, None

    @property
    def lenstronomy_ID(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        return ['SPL_CORE']

    @property
    def z_eval(self):
        """
        Returns the redshift at which to evalate the concentration-mass relation
        """
        if not hasattr(self, '_zeval'):

            if 'evaluate_mc_at_zlens' in self._args.keys() and self._args['evaluate_mc_at_zlens']:
                self._zeval = self.z
            else:
                self._zeval = self.z_infall

        return self._zeval

    @property
    def profile_args(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_profile_args'):
            if 'c' not in list(self._args.keys()):
                concentration = self._concentration_class.nfw_concentration(self.mass, self.z_eval)
            else:
                concentration = self._args['c']
            gamma = self._args['log_slope_halo']
            x_core_halo = self._args['x_core_halo']
            self._profile_args = (concentration, gamma, x_core_halo)

        return self._profile_args


class PowerLawFieldHalo(PowerLawSubhalo):
    """
    Class that defines a power law halo in the field
    """
    @property
    def profile_args(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_profile_args'):

            concentration = self._concentration_class.nfw_concentration(self.mass, self.z)
            gamma = self._args['log_slope_halo']
            x_core_halo = self._args['x_core_halo']
            self._profile_args = (concentration, gamma, x_core_halo)

        return self._profile_args

    @property
    def z_eval(self):
        """
        Returns the redshift at which to evaluate the concentration-mass relation
        """
        return self.z

class GlobularCluster(Halo):

    def __init__(self, mass, x, y, z, lens_cosmo_instance, args, unique_tag):
        """
        A mass profile model for a globular cluster
        :param mass: the total mass of the GC
        :param x: x coordinate [arcsec]
        :param y: y coordinate [arcsec]
        :param z: redshift
        :param lens_cosmo_instance: an instance of LensCosmo class
        :param args: keyword arguments for the profile, must contain 'gamma', 'gc_size_lightyear', 'gc_concentration'
        which are the logarithmic profile slope outside the core, the size of the GC in lightyears
        (the mass is defined as the mass inside a sphere with this radius), and the 'concentration' which is the ratio
         of the gc core size to total size
        :param unique_tag:
        """
        self._prof = SPLCORE()
        self._lens_cosmo = lens_cosmo_instance
        mdef = 'SPL_CORE'
        is_subhalo = False
        super(GlobularCluster, self).__init__(mass, x, y, None, mdef, z, is_subhalo,
                                              lens_cosmo_instance, args, unique_tag,
                                              fixed_position=True)

    def density_profile_2d_lenstronomy(self, r):
        """

        :param r:
        :return:
        """
        rho0, gc_size, gamma, r_core = self.profile_args
        return self._prof.density_2d(r, 0.0, rho0, r_core, gamma)

    def density_profile_3d_lenstronomy(self,r):
        """

        :param r:
        :return:
        """
        kwargs = self.lenstronomy_params[0][0]
        kpc_per_arcsec = self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)
        sigma_crit_mpc = self._lens_cosmo.get_sigma_crit_lensing(self.z, self._lens_cosmo.z_source)
        sigma_crit_kpc = sigma_crit_mpc * 1e-6
        factor = sigma_crit_kpc / kpc_per_arcsec
        return factor * self._prof.density_lens(r / kpc_per_arcsec,
                                       kwargs['sigma0'],
                                       kwargs['r_core'],
                                       kwargs['gamma'])

    @property
    def lenstronomy_params(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_lenstronomy_args'):
            kpc_per_lightyear = 0.000306
            kpc_per_arcsec = self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)
            gamma = self._args['gamma']
            gc_size_lightyear = self._args['gc_size_lightyear']
            gc_size_arcsec = gc_size_lightyear * kpc_per_lightyear / kpc_per_arcsec
            r_core_arcsec = gc_size_arcsec/self._args['gc_concentration']
            sigma_crit_mpc = self._lens_cosmo.get_sigma_crit_lensing(self.z, self._lens_cosmo.z_source)
            sigma_crit_arcsec = sigma_crit_mpc * (0.001 * kpc_per_arcsec) ** 2
            rho0 = self.mass / self._prof.mass_3d(gc_size_arcsec, sigma_crit_arcsec, r_core_arcsec, gamma)
            sigma0 = rho0 * r_core_arcsec
            self._lenstronomy_args = [{'sigma0': sigma0, 'gamma': gamma, 'center_x': self.x, 'center_y': self.y,
                                       'r_core': r_core_arcsec}]
        return self._lenstronomy_args, None

    @property
    def profile_args(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_profile_args'):

            kpc_per_lightyear = 0.000306
            gamma = self._args['gamma']
            gc_size_lightyear = self._args['gc_size_lightyear']
            c = self._args['gc_concentration']
            gc_size_kpc = gc_size_lightyear * kpc_per_lightyear
            r_core_kpc = gc_size_kpc / c
            rho0 = self.mass / self._prof.mass_3d(gc_size_kpc, 1.0, r_core_kpc, gamma)
            self._profile_args = (rho0, gc_size_kpc, gamma, r_core_kpc)
        return self._profile_args

    @property
    def lenstronomy_ID(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        return ['SPL_CORE']

    @property
    def z_eval(self):
        """
        Returns the redshift at which to evalate the concentration-mass relation
        """
        return self.z
