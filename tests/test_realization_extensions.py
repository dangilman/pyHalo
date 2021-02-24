import pytest
from pyHalo.single_realization import SingleHalo
from pyHalo.realization_extensions import RealizationExtensions
import numpy.testing as npt
import numpy as np

class TestRealizationExtensions(object):

    def test_core_collapsed_halo(self):

        single_halo = SingleHalo(10 ** 8, 0.5, -0.1, 100, 'TNFW', 0.5, 0.5, 1.5, subhalo_flag=True)
        ext = RealizationExtensions(single_halo)
        new = ext.add_core_collapsed_halos([0])
        lens_model_list = new.lensing_quantities()[0]
        npt.assert_string_equal(lens_model_list[0], 'PJAFFE')

    def core_collapsed_halos(self):

        def timescalefunction_short(rhos, rs, v):
            return 1e-9
        def timescalefunction_long(rhos, rs, v):
            return 1e9

        single_halo = SingleHalo(10 ** 8, 0.5, -0.1, 100, 'TNFW', 0.5, 0.5, 1.5, subhalo_flag=True)
        ext = RealizationExtensions(single_halo)
        vfunc = lambda x: 4 / np.sqrt(3.1459)

        indexes = ext.find_core_collapsed_halos(timescalefunction_short, vfunc)
        npt.assert_equal(True, 0 in indexes)
        indexes = ext.find_core_collapsed_halos(timescalefunction_long, vfunc)
        npt.assert_equal(False, 0 in indexes)

if __name__ == '__main__':
    pytest.main()
