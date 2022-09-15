from pyHalo.Halos.halo_base import Halo
import numpy as np

class coreNFWFieldHalo(Halo):

    """
    The main class for an NFW field halo profile without truncation

    See the base class in Halos/halo_base.py for the required routines for any instance of a Halo class
    """

    def __init__(self, mass, x, y, r3d, mdef, z,
                 sub_flag, lens_cosmo_instance, args, unique_tag):

        """
        See documentation in base class (Halos/halo_base.py)
        """
        self._lens_cosmo = lens_cosmo_instance
        super(coreNFWFieldHalo, self).__init__(mass, x, y, r3d, mdef, z, sub_flag,
                                            lens_cosmo_instance, args, unique_tag)

    @property
    def z_eval(self):
        """
        Returns the redshift at which to evalate the concentration-mass relation
        """

        return self.z

    @property
    def lenstronomy_ID(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """

        return ['CNFW']

    @property
    def lenstronomy_params(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """

        (concentration, r_core_units_rs) = self.profile_args
        Rs_angle, theta_Rs = self._lens_cosmo.nfw_physical2angle(self.mass, concentration, self.z)
        r_core = r_core_units_rs * Rs_angle
        x, y = np.round(self.x, 4), np.round(self.y, 4)
        Rs_angle = np.round(Rs_angle, 10)
        theta_Rs = np.round(theta_Rs, 10)
        r_core = np.round(r_core, 10)

        kwargs = [{'alpha_Rs': self._rescale_norm * theta_Rs, 'Rs': Rs_angle,
                  'center_x': x, 'center_y': y, 'r_core': r_core}]

        return kwargs, None

    @property
    def c(self):
        """
        Computes the halo concentration (once)
        """

        if not hasattr(self, '_c'):
            self._c = self._lens_cosmo.NFW_concentration(self.mass,
                                                         self.z_eval,
                                                         self._args['mc_model'],
                                                         self._args['mc_mdef'],
                                                         self._args['log_mc'],
                                                         self._args['c_scatter'],
                                                         self._args['c_scatter_dex'],
                                                         self._args['kwargs_suppression'],
                                                         self._args['suppression_model'])
        return self._c

    @property
    def profile_args(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_profile_args'):
            concentration = self.c
            """
            TODO: make give the ability pass in a function to compute core radius
            """
            r_core_units_rs = self._args['r_core_units_rs']
            self._profile_args = (concentration, r_core_units_rs)

        return self._profile_args

class coreNFWSubhalo(coreNFWFieldHalo):

    """
    The main class for an NFW subhalo profile without truncation

    See the base class in Halos/halo_base.py for the required routines for any instance of a Halo class
    """

    @property
    def z_eval(self):
        """
        Returns the redshift at which to evalate the concentration-mass relation
        """
        if not hasattr(self, '_zeval'):

            if self._args['evaluate_mc_at_zlens']:
                self._zeval = self.z
            else:
                self._zeval = self.z_infall

        return self._zeval
