from pyHalo.Halos.halo_base import Halo
import numpy as np
from lenstronomy.LensModel.Profiles.pseudo_double_powerlaw import PseudoDoublePowerlaw


class SIS(Halo):
    """
    The base class for an SIS object; this is intended to represent LMC-like objects with enough luminous material
    that they do not have NFW profiles

    This class is intended instantiated from an NFW halo class
    """
    def __init__(self, nfw_halo):
        """

        :param nfw_halo: an instance of TNFW, or TNFWCHalo halo classes
        """
        self._lens_cosmo = nfw_halo.lens_cosmo
        mdef = 'SIS'
        self._nfw_halo = nfw_halo
        super(SIS, self).__init__(mass=nfw_halo.mass,
                                  x=nfw_halo.x,
                                  y=nfw_halo.y,
                                  r3d=nfw_halo.r3d,
                                  mdef=mdef,
                                  z=nfw_halo.z,
                                  sub_flag=nfw_halo.is_subhalo,
                                  lens_cosmo_instance=nfw_halo.lens_cosmo,
                                  args=nfw_halo._args,
                                  unique_tag=nfw_halo.unique_tag,
                                  fixed_position=nfw_halo.fixed_position)

    @property
    def profile_args(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_profile_args'):
            vmax = self._nfw_halo.vmax_nfw
            velocity_dispersion = 1.0 * vmax
            self._profile_args = (velocity_dispersion)
        return self._profile_args

    @property
    def bound_mass(self):
        if self.is_subhalo:
            return self._nfw_halo.bound_mass
        else:
            raise Exception('field halos do not have bound mass attribute')

    @property
    def lenstronomy_ID(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        return ['SIS']

    @property
    def lenstronomy_params(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_kwargs_lenstronomy'):
            sigma_v = self.profile_args
            theta_E = self._lens_cosmo.thetaE_from_sigma(self.z, sigma_v)
            #mass3d = sis_lenstronomy.mass_3d_lens(rs, thetaE) * nfw_field_halo.lens_cosmo.sigmacrit
            x, y = np.round(self.x, 4), np.round(self.y, 4)
            kwargs = [{'theta_E': theta_E, 'center_x': x,  'center_y': y}]
            self._kwargs_lenstronomy = kwargs
        return self._kwargs_lenstronomy, None

class MassiveGalaxy(Halo):
    _pseudo_nfw = True
    def __init__(self, nfw_halo):
        """

        :param nfw_halo: an instance of TNFW, or TNFWCHalo halo classes
        """
        self._lens_cosmo = nfw_halo.lens_cosmo
        mdef = 'GNFW'
        self._nfw_halo = nfw_halo
        self._prof = PseudoDoublePowerlaw()
        halo_args = {'gamma_inner': 2.0, 'gamma_outer': 3.1, 'x_match': 'c'}
        super(MassiveGalaxy, self).__init__(mass=nfw_halo.mass,
                                  x=nfw_halo.x,
                                  y=nfw_halo.y,
                                  r3d=nfw_halo.r3d,
                                  mdef=mdef,
                                  z=nfw_halo.z,
                                  sub_flag=nfw_halo.is_subhalo,
                                  lens_cosmo_instance=nfw_halo.lens_cosmo,
                                  args=halo_args,
                                  unique_tag=nfw_halo.unique_tag,
                                  fixed_position=nfw_halo.fixed_position)

    @property
    def c(self):
        """
        Computes the halo concentration (once)
        """
        if not hasattr(self, '_c'):
            self._c = self._nfw_halo.c
        return self._c

    @property
    def lenstronomy_params(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_lenstronomy_args'):
            (concentration, gamma_inner, gamma_outer) = self.profile_args
            _, rs, r200 = self.nfw_params
            kpc_per_arcsec = self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)
            rs_arcsec = rs / kpc_per_arcsec
            alpha_Rs = self._alphaRs()
            x, y = np.round(self.x, 4), np.round(self.y, 4)
            rs_arcsec = np.round(rs_arcsec, 10)
            alpha_Rs = np.round(alpha_Rs, 10)
            self._lenstronomy_args = [{'alpha_Rs': alpha_Rs, 'Rs': rs_arcsec,
                                       'gamma_inner': gamma_inner,
                                       'center_x': x, 'center_y': y,
                                      'gamma_outer': gamma_outer}]
        return self._lenstronomy_args, None

    @property
    def lenstronomy_ID(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        return ['PSEUDO_DPL']

    @property
    def profile_args(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_profile_args'):

            concentration = self._nfw_halo.c
            _ = self._nfw_halo.profile_args
            gamma_inner = self._args['gamma_inner']
            gamma_outer = self._args['gamma_outer']
            self._profile_args = (concentration, gamma_inner, gamma_outer)

        return self._profile_args

    def _alphaRs(self):
        """
        Calculates the density normalization that conserves mass within x_match
        :return:
        """
        if not hasattr(self, '_rho0_norm'):
            kpc_per_arcsec = self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)
            m_nfw = self._nfw_halo.mass_3d('r200')
            _, rs_kpc, r200_kpc = self._nfw_halo.nfw_params
            rs_arcsec = rs_kpc / kpc_per_arcsec
            r_match_arcsec = r200_kpc / kpc_per_arcsec
            gamma_inner = self._args['gamma_inner']
            gamma_outer = self._args['gamma_outer']
            sigma_crit_mpc = self._lens_cosmo.get_sigma_crit_lensing(self.z, self._lens_cosmo.z_source)
            sigma_crit_arcsec = sigma_crit_mpc * (0.001 * kpc_per_arcsec) ** 2
            self._alphaRs = m_nfw / self._prof.mass_3d_lens(r_match_arcsec,
                                                            rs_arcsec,
                                                            1.0,
                                                            gamma_inner,
                                                            gamma_outer) / sigma_crit_arcsec
        return self._alphaRs





