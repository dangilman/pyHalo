from pyHalo.Halos.cosmo_profiles import CosmoMassProfiles
import numpy

class PJAFFE(CosmoMassProfiles):

    def pjaffe_physical2angle(self, M, r_core, r_s):

        sigma = 1 * (M * 10 ** -8)

        return sigma
