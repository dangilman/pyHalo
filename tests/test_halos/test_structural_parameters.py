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

    def test_nfw(self):

        M = 10**8
        z = 0.5

        c1 = self.lens_cosmo.NFW_concentration(M, z, scatter=False)
        c2 = 2*c1
        rho1, rs1, r2001 = self.lens_cosmo.NFW_params_physical(M, c1, z)
        rho2, rs2, r2002 = self.lens_cosmo.NFW_params_physical(M, c2, z)

        m200_1 = 4 * np.pi * rs1 ** 3 * rho1 * (np.log(1 + c1) - c1 / (1 + c1))
        m200_2 = 4 * np.pi * rs2 ** 3 * rho2 * (np.log(1 + c2) - c2 / (1 + c2))
        npt.assert_almost_equal(m200_1/m200_2)
