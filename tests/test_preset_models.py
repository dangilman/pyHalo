from pyHalo.preset_models import WDMLovell2020, CDM
import numpy.testing as npt
import pytest

class TestPresetModels(object):

    def test_CDM(self):

        realization_cdm = CDM(0.5, 1.5)
        npt.assert_equal(len(realization_cdm.rendering_classes), 2)

    def test_WDMLovell20(self):

        realization_wdm = WDMLovell2020(0.5, 1.5, 8.)
        npt.assert_equal(len(realization_wdm.rendering_classes), 2)

if __name__ == '__main__':
    pytest.main()
