class PrimordialBlackHole(object):

    @property
    def halo_parameters(self):
        return []

    @property
    def physical_params(self):
        if not hasattr(self, '_rho_sub') or not hasattr(self, '_rs_sub'):
            self._rho_sub, self._rs_sub, _ = self._halo_class.cosmo_prof.NFW_params_physical(self._halo_class.mass,
                                                                                             self.concentration,
                                                                                             self.halo_redshift_eval)

        return None

