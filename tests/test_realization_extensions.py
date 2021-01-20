import pytest
from pyHalo.single_realization import SingleHalo
from pyHalo.realization_extensions import RealizationExtensions
import numpy.testing as npt


class TestRealizationExtensions(object):

    def test_core_collapsed_halo(self):

        single_halo = SingleHalo(10 ** 8, 0.5, -0.1, 100, 'TNFW', 0.5, 0.5, 1.5, subhalo_flag=True)
        ext = RealizationExtensions(single_halo)
        new = ext.add_core_collapsed_subhalos(1.)
        lens_model_list = new.lensing_quantities()[0]
        npt.assert_string_equal(lens_model_list[0], 'PJAFFE')

if __name__ == '__main__':
    pytest.main()
