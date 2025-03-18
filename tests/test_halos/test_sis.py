import numpy.testing as npt
from pyHalo.Halos.HaloModels.sis import SIS
from pyHalo.Halos.HaloModels.TNFW import TNFWFieldHalo
from pyHalo.Halos.concentration import ConcentrationDiemerJoyce
from pyHalo.Halos.tidal_truncation import TruncationRoche
from astropy.cosmology import FlatLambdaCDM
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Cosmology.cosmology import Cosmology
from lenstronomy.LensModel.Profiles.sis import SIS as SISLenstronomy
import pytest
import numpy as np


class TestSISHalo(object):

    def setup_method(self):
        astropy = FlatLambdaCDM(70.0, 0.3)
        cosmo = Cosmology(astropy_instance=astropy)
        self.zhalo = 0.5
        self.zsource = 2.0
        self.lens_cosmo = LensCosmo(self.zhalo, self.zsource, cosmo)
        self.truncation_class = TruncationRoche(None, 100000000.0)
        self.concentration_class = ConcentrationDiemerJoyce(self.lens_cosmo, scatter=False)

    def test_lenstronomy_params(self):

        m = 10 ** 11.3
        x = 0.5
        y = 1.0
        r3d = 100
        unique_tag = 1.0
        is_subhalo = False
        nfw_field_halo = TNFWFieldHalo(m, x, y, r3d, self.zhalo, is_subhalo, self.lens_cosmo, {},
                                       self.truncation_class, self.concentration_class, unique_tag)

        sis = SIS(nfw_field_halo)
        vdis = sis.profile_args
        D_ds = nfw_field_halo.lens_cosmo.cosmo.D_A(0.5, 2.0)
        D_s = nfw_field_halo.lens_cosmo.cosmo.D_A(0., 2.0)
        arcsec = 206265
        c = 299792.5  # in km/sec
        thetaE = 4*np.pi*vdis**2/c**2 * D_ds / D_s * arcsec
        npt.assert_almost_equal(thetaE, sis.lenstronomy_params[0][0]['theta_E'])

    def test_sis_mass(self):

        m = 10 ** 11.3
        x = 0.5
        y = 1.0
        r3d = 100
        unique_tag = 1.0
        is_subhalo = False
        nfw_field_halo = TNFWFieldHalo(m, x, y, r3d, self.zhalo, is_subhalo, self.lens_cosmo, {},
                                       self.truncation_class, self.concentration_class, unique_tag)

        sis = SIS(nfw_field_halo)
        thetaE = sis.lenstronomy_params[0][0]['theta_E']
        _, rs, r200 = nfw_field_halo.nfw_params

        sis_lenstronomy = SISLenstronomy()
        mass3d = sis_lenstronomy.mass_3d_lens(rs, thetaE) * nfw_field_halo.lens_cosmo.sigmacrit
        npt.assert_almost_equal(mass3d/nfw_field_halo.mass, 2.05, 2)

if __name__ == '__main__':
    pytest.main()

