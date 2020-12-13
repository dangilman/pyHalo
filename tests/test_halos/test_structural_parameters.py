import numpy.testing as npt
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Halos.structural_parameters import HaloStructure
import numpy as np
from astropy.cosmology import FlatLambdaCDM
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
        self.astropy = self.cosmo.astropy
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
        npt.assert_almost_equal(m200_1/m200_2, 1)

        rho0c = 2.77536627e11

        npt.assert_almost_equal(rho0c/self.lens_cosmo.rhoc, 1, 3)

        rho0 = 1.576764e+16
        rs = 0.000711265
        r200 = 0.00640138

        (rho0_2, rs_2, r200_2) = self.lens_cosmo._nfwParam_physical_Mpc(10**8, 9, 0.5)
        npt.assert_almost_equal(rho0/rho0_2, 1, 2)
        npt.assert_almost_equal(rs/rs_2, 1, 2)
        npt.assert_almost_equal(r200/r200_2, 1, 2)

        (rho0_3, rs_3, r200_3) = self.lens_cosmo.NFW_params_physical(10 ** 8, 9, 0.5)
        npt.assert_almost_equal(rho0_3*1000 ** 3 / rho0_2, 1, 2)
        npt.assert_almost_equal(rs_3 * 0.001 / rs_2, 1, 2)
        npt.assert_almost_equal(r200_3 * 0.001 / r200_2, 1, 2)

        rs, alpha_rs = 0.11527, 0.000693
        rs_2, alpha_rs_2 = self.lens_cosmo.nfw_physical2angle(10**8, 9, 0.5)
        npt.assert_almost_equal(rs/rs_2, 1, 3)
        npt.assert_almost_equal(alpha_rs / alpha_rs_2, 1, 2)



if __name__ == '__main__':
     pytest.main()
