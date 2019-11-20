import numpy.testing as npt
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Halos.structural_parameters import HaloStructure
import numpy as np
import pytest

class TestStructuralParameters(object):

    def setup(self):
        H0 = 70
        omega_baryon = 0.03
        omega_DM = 0.25
        sigma8 = 0.82
        curvature = 'flat'
        ns = 0.9608
        cosmo_params = {'H0': H0, 'Om0': omega_baryon + omega_DM, 'Ob0': omega_baryon,
                        'sigma8': sigma8, 'ns': ns, 'curvature': curvature}
        self._dm, self._bar = omega_DM, omega_baryon
        self.cosmo = Cosmology(cosmo_kwargs=cosmo_params)

        zlens, zsource = 0.5, 1.5
        self.lens_cosmo = LensCosmo(zlens, zsource, self.cosmo)

        self.structure = HaloStructure(self.lens_cosmo)
