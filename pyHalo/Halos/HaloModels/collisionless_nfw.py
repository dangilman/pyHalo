from pyHalo.Halos.HaloModels.base import MainSubhaloBase, FieldHaloBase

class NFWFieldHalo(FieldHaloBase):

    @property
    def halo_parameters(self):
        return [self.concentration]

class NFWMainSubhalo(MainSubhaloBase):

    @property
    def halo_parameters(self):
        return [self.concentration]

class TNFWFieldHalo(FieldHaloBase):

    @property
    def halo_parameters(self):

        return [self.concentration, self.truncation_radius]

class TNFWMainSubhalo(MainSubhaloBase):

    @property
    def halo_parameters(self):
        return [self.concentration, self.truncation_radius]

    @property
    def truncation_radius(self):

        r_t = self._halo_class.lens_cosmo.truncation_roche(self._halo_class.mass, self._halo_class.r3d,
                                                               self._halo_class._args['RocheNorm'],
                                                               self._halo_class._args['RocheNu'])

        return r_t




