"""
These functions define some preset halo mass function models that can be specified instead of individually specifying
a bunch of keyword arguments. This is intended for a quick and easy user interface. The definitions shown here can also
act as a how-to guide if one wants to explore more complicated descriptions of the halo mass function, as the models
presented here show what each keyword argument accepted by pyHalo does.
"""

__all__ = ['preset_model_from_name']

def preset_model_from_name(name):
    """
    Retruns a preset_model function from a string
    :param name: the name of the preset model, should be the name of a function in this file
    :return: the function
    """
    if name == 'CDM':
        from pyHalo.PresetModels.cdm import CDM
        return CDM
    elif name == 'WDM':
        from pyHalo.PresetModels.wdm import WDM
        return WDM
    elif name == 'SIDM_core_collapse':
        from pyHalo.PresetModels.sidm import SIDM_core_collapse
        return SIDM_core_collapse
    elif name == 'SIDM_parametric':
        from pyHalo.PresetModels.sidm import SIDM_parametric
        return SIDM_parametric
    elif name == 'SIDM_parametric_fixedbins':
        from pyHalo.PresetModels.sidm import SIDM_parametric_fixedbins
        return SIDM_parametric_fixedbins
    elif name == 'ULDM':
        from pyHalo.PresetModels.uldm import ULDM
        return ULDM
    elif name == 'CDMEmulator':
        from pyHalo.PresetModels.external import CDMFromEmulator
        return CDMFromEmulator
    elif name == 'WDM_mixed':
        from pyHalo.PresetModels.wdm import WDM_mixed
        return WDM_mixed
    elif name == 'WDMGeneral':
        from pyHalo.PresetModels.wdm import WDMGeneral
        return WDMGeneral
    elif name == "DMGalacticus":
        from pyHalo.PresetModels.external import DMFromGalacticus
        return DMFromGalacticus
    elif name == 'CDM_plus_BH':
        from pyHalo.PresetModels.mbh import CDM_plus_BH
        return CDM_plus_BH
    else:
        raise Exception('preset model '+ str(name)+' not recognized!')



















