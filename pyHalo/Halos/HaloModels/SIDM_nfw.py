from pyHalo.Halos.HaloModels.base import MainSubhaloBase, FieldHaloBase
from pyHalo.Scattering.sidm_interp import logrho
from pyHalo.Halos.halo_util import *

class truncatedSIDMMainSubhalo(MainSubhaloBase):

    @property
    def physical_args(self):

        if not hasattr(self, '_rho_sub') or not hasattr(self, '_rs_sub'):
            self._rho_sub, self._rs_sub, _ = self._halo_class.cosmo_prof.NFW_params_physical(self._halo_class.mass,
                                                                                             self.concentration,
                                                                                             self.halo_redshift_eval)

        return {'rho_s': self._rho_sub, 'rs': self._rs_sub, 'c': self.concentration,
                'rt': self.truncation_radius, 'rc': self.core_radius*self._rs_sub}

    @property
    def halo_parameters(self):

        return [self.concentration, self.truncation_radius, self.core_radius]

    @property
    def truncation_radius(self):

        condition_1 = self._halo_class._args['truncation_routine'] == \
                      'mean_NFWhost'
        condition_2 = self._halo_class._args['truncation_routine'] == \
                      'mean_ISOhost'

        if condition_1 or condition_2:

            subhalo_density_function = cored_rho_nfw
            rho_sub, rs_sub, _ = self._halo_class.cosmo_prof.NFW_params_physical(self._halo_class.mass,
                                                                                 self.concentration,
                                                                                 self.halo_redshift_eval)

            subhalo_args = ((rs_sub, rho_sub), self.core_radius * rs_sub)
            initial_guess = 5*rs_sub
            r_ein_kpc = self._halo_class._args['R_ein_main'] * \
                        self._halo_class.cosmo_prof._kpc_per_arcsec_zlens

            if condition_1:

                r_t = self._halo_class.cosmo_prof.truncation_mean_density((self._halo_class.mass, self.concentration,
                                                                          self._halo_class._args['parent_m200'],
                                                                          self._halo_class.pericenter, self.halo_redshift_eval,
                                                                          self._halo_class.cosmo_prof.z_lens,
                                                                          r_ein_kpc, subhalo_density_function,
                                                                          subhalo_args, initial_guess))

            elif condition_2:

                sigmacrit_kpc = self._halo_class.cosmo_prof.epsilon_crit_kpc

                r_t = self._halo_class.cosmo_prof.truncation_mean_density_isothermal_host((self._halo_class.mass,
                                                                                          self.concentration,
                                                                                          self._halo_class._args[
                                                                                              'parent_m200'],
                                                                                          self._halo_class.pericenter,
                                                                                          self.halo_redshift_eval,
                                                                                          self._halo_class.cosmo_prof.z_lens,
                                                                                          r_ein_kpc, sigmacrit_kpc,
                                                                                          subhalo_density_function,
                                                                                          subhalo_args, initial_guess))

        else:
            r_t = self._halo_class.cosmo_prof.truncation_roche((self._halo_class.mass, self._halo_class.pericenter,
                                                               self._halo_class.z,
                                                               self._halo_class._args['RocheNorm'],
                                                               self._halo_class._args['RocheNu']))


        return r_t

    @property
    def core_radius(self):

        if 'core_ratio' in self._halo_class._args.keys():
            if 'SIDMcross' in self._halo_class._args.keys():
                raise Exception('You have specified both core_ratio and SIDMcross arguments. '
                                'You should pick one or the other')
            core_ratio = self._halo_class._args['core_ratio']

        else:

            cmean = self._halo_class.cosmo_prof.NFW_concentration(self._halo_class.mass, self._halo_class.z,
                                                                  scatter=False)
            rho_mean, rs_mean, _ = self._halo_class.cosmo_prof.NFW_params_physical(self._halo_class.mass,
                                                                                   cmean, self._halo_class.z)

            if 'halo_age' not in self._halo_class._args.keys():
                halo_age = self._halo_class.cosmo_prof.astropy_cosmo.halo_age(self._halo_class.z)
            else:
                halo_age = self._halo_class._args['halo_age']

            zeta = self._halo_class._args['SIDMcross'] * halo_age

            rho_sidm = 10 ** logrho(self._halo_class.mass, self._halo_class.z, zeta, cmean,
                                    self.concentration, self._halo_class._args['vpower'])

            core_ratio = rho_mean * rho_sidm ** -1

        return core_ratio

class truncatedSIDMFieldHalo(FieldHaloBase):

    @property
    def physical_args(self):

        if not hasattr(self, '_rho_sub') or not hasattr(self, '_rs_sub'):
            self._rho_sub, self._rs_sub, _ = self._halo_class.cosmo_prof.NFW_params_physical(self._halo_class.mass,
                                                                                             self.concentration,
                                                                                             self.halo_redshift_eval)

        return {'rho_s': self._rho_sub, 'rs': self._rs_sub, 'c': self.concentration,
                'rt': self.truncation_radius, 'rc': self.core_radius*self._rs_sub}

    @property
    def halo_parameters(self):

        return [self.concentration, self.truncation_radius, self.core_radius]

    @property
    def core_radius(self):


        if 'core_ratio' in self._halo_class._args.keys():
            if 'SIDMcross' in self._halo_class._args.keys():
                raise Exception('You have specified both core_ratio and SIDMcross arguments. '
                                'You should pick one or the other')
            core_ratio = self._halo_class._args['core_ratio']

        else:

            cmean = self._halo_class.cosmo_prof.NFW_concentration(self._halo_class.mass, self._halo_class.z,
                                                                  scatter=False)
            rho_mean, rs_mean, _ = self._halo_class.cosmo_prof.NFW_params_physical(self._halo_class.mass,
                                                                                   cmean, self._halo_class.z)

            if 'halo_age' not in self._halo_class._args.keys():
                halo_age = self._halo_class.cosmo_prof.astropy_cosmo.halo_age(self._halo_class.z)
            else:
                halo_age = self._halo_class._args['halo_age']

            zeta = self._halo_class._args['SIDMcross'] * halo_age

            rho_sidm = 10 ** logrho(self._halo_class.mass, self._halo_class.z, zeta, cmean,
                                    self.concentration, self._halo_class._args['vpower'])

            core_ratio = rho_mean * rho_sidm ** -1

        return core_ratio
