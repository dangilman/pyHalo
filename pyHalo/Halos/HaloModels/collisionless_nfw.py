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




