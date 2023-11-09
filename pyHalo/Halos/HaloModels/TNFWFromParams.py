import numpy as np
from pyHalo.Halos.HaloModels.TNFW import TNFWSubhalo

class TNFWFromParams(TNFWSubhalo):
    """
    Creates a TNFW halo based on physical params.
    """

    KEY_RT = "r_trunc_kpc"
    KEY_RS = "rs"
    KEY_RHO_S = "rhos"
    KEY_RV = "rv"
    KEY_Z_INFALL = "z_infall"
    KEY_ID = "index"

    def __init__(self, mass, x_kpc, y_kpc, r3d, z,sub_flag,
                 lens_cosmo_instance, args, unique_tag=None):
        """
        Defines a TNFW subhalo with physical params r_trunc_kpc, rs, rhos passed in the args argument
        """

        self._lens_cosmo = lens_cosmo_instance
        self._kpc_per_arcsec_at_z = self._lens_cosmo.cosmo.kpc_proper_per_asec(z)
        x = x_kpc / self._kpc_per_arcsec_at_z
        y = y_kpc / self._kpc_per_arcsec_at_z
        keys_physical = (self.KEY_RV,self.KEY_RS,self.KEY_RHO_S,self.KEY_RV,self.KEY_RT)
        self._params_physical = {key:args[key] for key in keys_physical}
        self._c = self._params_physical[self.KEY_RV] / self._params_physical[self.KEY_RS]
        self.id = args.get(self.KEY_ID)
        self._z_infall = args.get(self.KEY_Z_INFALL)
        super(TNFWFromParams, self).__init__(mass,x,y,r3d,z,sub_flag,lens_cosmo_instance,args,None,None,unique_tag)

    def density_profile_3d(self, r, params_physical=None):
        """
        Computes the 3-D density profile of the halo
        :param r: distance from center of halo [kpc]
        :return: the density profile in units M_sun / kpc^3
        """

        _params = self._params_physical if params_physical is None else params_physical

        r_t = _params[self.KEY_RT]
        r_s = _params[self.KEY_RS]
        rho_s = _params[self.KEY_RHO_S]

        n = 1

        x = r / r_s
        tau = r_t / r_s

        density_nfw = (rho_s / ((x)*(1+x)**2))

        return density_nfw * (tau**2 / (tau**2 + x**2))**n


    @property
    def profile_args(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_profile_args'):
            truncation_radius_kpc = self.params_physical[self.KEY_RT]
            self._profile_args = (self.c, truncation_radius_kpc)
        return self._profile_args


    @property
    def lenstronomy_params(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        KPC_TO_MPC = 1E-3

        if not hasattr(self, '_kwargs_lenstronomy'):

            r_t = self.params_physical[self.KEY_RT]
            r_s = self.params_physical[self.KEY_RS]
            rho_s = self.params_physical[self.KEY_RHO_S]

            Rs_angle, theta_Rs = self.lens_cosmo.nfw_physical2angle_fromNFWparams(rho_s *1 / KPC_TO_MPC**3,r_s * KPC_TO_MPC,self.z)

            x, y = np.round(self.x, 4), np.round(self.y, 4)

            Rs_angle = np.round(Rs_angle, 10)
            theta_Rs = np.round(theta_Rs, 10)
            r_trunc_arcsec = r_t / self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)

            kwargs = [{'alpha_Rs': self._rescale_norm * theta_Rs, 'Rs': Rs_angle,
                      'center_x': x, 'center_y': y, 'r_trunc': r_trunc_arcsec}]

            self._kwargs_lenstronomy = kwargs

        return self._kwargs_lenstronomy, None
