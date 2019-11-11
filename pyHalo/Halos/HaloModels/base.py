class MainSubhaloBase(object):

    def __init__(self, halo_class):

        self._halo_class = halo_class

    @property
    def concentration(self):

        if self._halo_class._args['evaluate_mc_at_zlens']:
            z_eval = self._halo_class.z
        else:
            z_eval = self._halo_class.get_z_infall()

        return self._halo_class.cosmo_prof.NFW_concentration(self._halo_class.mass, z_eval,
                                                 logmhm=self._halo_class._args['log_m_break'],
                                                 c_scale=self._halo_class._args['c_scale'],
                                                 c_power=self._halo_class._args['c_power'],
                                                 scatter=self._halo_class._args['c_scatter'],
                                                 model=self._halo_class._args['mc_model'])

    @property
    def truncation_radius(self):
        return self._halo_class.cosmo_prof.truncation_roche(self._halo_class.mass, self._halo_class.r3d, self._halo_class.z,
                                                self._halo_class._args['RocheNorm'],
                                                self._halo_class._args['RocheNu'])


class FieldHaloBase(object):

    def __init__(self, halo_class):
        self._halo_class = halo_class

    @property
    def concentration(self):
        return self._halo_class.cosmo_prof.NFW_concentration(self._halo_class.mass, self._halo_class.z,
                                                             logmhm=self._halo_class._args['log_m_break'],
                                                             c_scale=self._halo_class._args['c_scale'],
                                                             c_power=self._halo_class._args['c_power'],
                                                             scatter=self._halo_class._args['c_scatter'],
                                                             model=self._halo_class._args['mc_model'])

    @property
    def truncation_radius(self):
        return self._halo_class.cosmo_prof.LOS_truncation(self._halo_class.mass, self._halo_class.z,
                                                          self._halo_class._args['LOS_truncation_factor'])

