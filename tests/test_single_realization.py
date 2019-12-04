import numpy.testing as npt
from pyHalo.Massfunc.parameterizations import BrokenPowerLaw, PowerLaw
from pyHalo.Cosmology.lensing_mass_function import LensingMassFunction
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.single_realization import Realization, RealizationFast
from pyHalo.Massfunc.los import LOSPowerLaw
import pytest
import numpy as np
from copy import copy

class TestSingleRealization(object):

    def setup(self):

        masses = [10**8, 10**8]
        x = [0.3, 0.0]
        y = [-2, 0.]
        r2d = [0.3, 0.5]
        r3d = [20, 20]
        mdefs = ['TNFW']*2
        z = [0.5, 1.5]
        subhalo_flag = [False, False]
        self.zlens, self.zsource = 0.5, 1.5
        cone_opening_angle = 6
        self.realization = RealizationFast(masses, x, y, r2d, r3d, mdefs, z, subhalo_flag, self.zlens, self.zsource,
                 cone_opening_angle, log_mlow=6, log_mhigh=10, mass_sheet_correction=False)

    def test_shift_to_source(self):

        source_x, source_y = 0, 0
        realization_shifted = self.realization.shift_background_to_source(source_x, source_y)
        for halo, halo_shifted in zip(self.realization.halos, realization_shifted.halos):

            assert halo.x == halo_shifted.x
            assert halo.y == halo_shifted.y

        source_x, source_y = 1, 0
        realization_shifted = self.realization.shift_background_to_source(source_x, source_y)
        for i, (halo, halo_shifted) in enumerate(zip(self.realization.halos, realization_shifted.halos)):

            if i==0:
                assert halo.x == halo_shifted.x
                assert halo.y == halo_shifted.y
            else:
                assert halo.x != halo_shifted.x
                assert halo_shifted.x == source_x
                assert halo.y == source_y

#t.test_mass_at_z()
#
# if __name__ == '__main__':
#     pytest.main()



