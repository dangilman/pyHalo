from pyHalo.Halos.HaloModels.base import MainSubhaloBase, FieldHaloBase
from pyHalo.Halos.halo_util import *

class NFWFieldHalo(FieldHaloBase):

    @property
    def halo_parameters(self):
        return [self.concentration]

    @property
    def physical_args(self):
        if not hasattr(self, '_rho_sub') or not hasattr(self, '_rs_sub'):
            self._rho_sub, self._rs_sub, _ = self._halo_class.cosmo_prof.NFW_params_physical(self._halo_class.mass,
                                                                                             self.concentration,
                                                                                             self.halo_redshift_eval)

        return {'rho_s': self._rho_sub, 'rs': self._rs_sub, 'c': self.concentration}

class NFWMainSubhalo(MainSubhaloBase):

    @property
    def halo_parameters(self):
        return [self.concentration]

    @property
    def physical_args(self):
        if not hasattr(self, '_rho_sub') or not hasattr(self, '_rs_sub'):
            self._rho_sub, self._rs_sub, _ = self._halo_class.cosmo_prof.NFW_params_physical(self._halo_class.mass,
                                                                                             self.concentration,
                                                                                             self.halo_redshift_eval)

        return {'rho_s': self._rho_sub, 'rs': self._rs_sub, 'c': self.concentration}

class TNFWFieldHalo(FieldHaloBase):

    @property
    def halo_parameters(self):

        return [self.concentration, self.truncation_radius]

    @property
    def physical_args(self):
        if not hasattr(self, '_rho_sub') or not hasattr(self, '_rs_sub'):
            self._rho_sub, self._rs_sub, _ = self._halo_class.cosmo_prof.NFW_params_physical(self._halo_class.mass,
                                                                                             self.concentration,
                                                                                             self.halo_redshift_eval)

        return {'rho_s': self._rho_sub, 'rs': self._rs_sub, 'c': self.concentration, 
                'rt': self.truncation_radius}

class TNFWMainSubhalo(MainSubhaloBase):
    
    @property
    def physical_args(self):
        
        if not hasattr(self, '_rho_sub') or not hasattr(self, '_rs_sub'):
            self._rho_sub, self._rs_sub, _ = self._halo_class.cosmo_prof.NFW_params_physical(self._halo_class.mass,
                                                                                 self.concentration,
                                                                                 self.halo_redshift_eval)
        
        return {'rho_s': self._rho_sub, 'rs': self._rs_sub, 'c': self.concentration, 
                'rt': self.truncation_radius}
        
    @property
    def halo_parameters(self):
        return [self.concentration, self.truncation_radius]

    @property
    def truncation_radius(self):

        condition_1 = self._halo_class._args['truncation_routine'] == \
                      'mean_NFWhost'
        condition_2 = self._halo_class._args['truncation_routine'] == \
                      'mean_ISOhost'

        if condition_1 or condition_2:

            subhalo_density_function = rho_nfw
            rho_sub, rs_sub, _ = self._halo_class.cosmo_prof.NFW_params_physical(self._halo_class.mass,
                                                                                 self.concentration,
                                                                                 self.halo_redshift_eval)

            initial_guess = 8 * rs_sub
            r_ein_kpc = self._halo_class._args['R_ein_main'] * \
                        self._halo_class.cosmo_prof._kpc_per_arcsec_zlens

            subhalo_args = (rs_sub, rho_sub)

            if condition_1:

                r_t = self._halo_class.cosmo_prof.truncation_mean_density((self._halo_class.mass, self.concentration,
                                                                      self._halo_class._args['parent_m200'],
                                                                      self._halo_class.r3d, self.halo_redshift_eval,
                                                                      self._halo_class.cosmo_prof.z_lens,
                                                                      r_ein_kpc, subhalo_density_function,
                                                                      subhalo_args, initial_guess))

            elif condition_2:

                sigmacrit_kpc = self._halo_class.cosmo_prof.epsilon_crit_kpc

                r_t = self._halo_class.cosmo_prof.truncation_mean_density_isothermal_host((self._halo_class.mass, self.concentration,
                                                self._halo_class._args['parent_m200'],self._halo_class.r3d,
                                                 self.halo_redshift_eval, self._halo_class.cosmo_prof.z_lens,
                                                    r_ein_kpc, sigmacrit_kpc, subhalo_density_function,
                                                                 subhalo_args, initial_guess))

        else:
            r_t = self._halo_class.cosmo_prof.truncation_roche((self._halo_class.mass, self._halo_class.r3d,
                                                               self._halo_class.z,
                                                               self._halo_class._args['RocheNorm'],
                                                               self._halo_class._args['RocheNu']))

        return r_t




