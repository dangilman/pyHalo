from pyHalo.Halos.HaloModels.base import MainSubhaloBase, FieldHaloBase
from pyHalo.Scattering.sidm_interp import logrho

class truncatedSIDMMainSubhalo(MainSubhaloBase):

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
                halo_age = self._halo_class.cosmo_prof.cosmo.halo_age(self._halo_class.z)
            else:
                halo_age = self._halo_class._args['halo_age']

            zeta = self._halo_class._args['SIDMcross'] * halo_age

            nfw_c = self.concentration
            rho_sidm = 10 ** logrho(self._halo_class.mass, self._halo_class.z, zeta, cmean,
                                    nfw_c, self._halo_class._args['vpower'])

            core_ratio = rho_mean * rho_sidm ** -1

        return core_ratio

class truncatedSIDMFieldHalo(FieldHaloBase):

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
                halo_age = self._halo_class.cosmo_prof.cosmo.halo_age(self._halo_class.z)
            else:
                halo_age = self._halo_class._args['halo_age']

            zeta = self._halo_class._args['SIDMcross'] * halo_age

            nfw_c = self.concentration
            rho_sidm = 10 ** logrho(self._halo_class.mass, self._halo_class.z, zeta, cmean,
                                    nfw_c, self._halo_class._args['vpower'])

            core_ratio = rho_mean * rho_sidm ** -1

        return core_ratio
