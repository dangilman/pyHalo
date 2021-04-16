from pyHalo.preset_models import WDMLovell2020, CDM, WDMGeneral
import numpy.testing as npt
import pytest

class TestPresetModels(object):

    def test_CDM(self):

        realization_cdm = CDM(0.5, 1.5)
        npt.assert_equal(len(realization_cdm.rendering_classes), 3)

    def test_WDMLovell20(self):

        realization_wdm = WDMLovell2020(0.5, 1.5, 8.)
        npt.assert_equal(len(realization_wdm.rendering_classes), 3)

    def test_WDMGeneral(self):

        realization_wdm = WDMGeneral(0.5, 1.5, 8.)
        npt.assert_equal(len(realization_wdm.rendering_classes), 3)

if __name__ == '__main__':
     pytest.main()
