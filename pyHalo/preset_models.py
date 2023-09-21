"""
These functions define some preset halo mass function models that can be specified instead of individually specifying
a bunch of keyword arguments. This is intended for a quick and easy user interface. The definitions shown here can also
act as a how-to guide if one wants to explore more complicated descriptions of the halo mass function, as the models
presented here show what each keyword argument accepted by pyHalo does.
"""
from pyHalo.PresetModels.cdm import CDM
from pyHalo.PresetModels.external import CDMFromEmulator, DMFromGalacticus
from pyHalo.PresetModels.sidm import SIDM_core_collapse
from pyHalo.PresetModels.uldm import ULDM
from pyHalo.PresetModels.wdm import WDM, WDM_mixed, WDMGeneral


__all__ = ['preset_model_from_name']

def preset_model_from_name(name):
    """
    Retruns a preset_model function from a string
    :param name: the name of the preset model, should be the name of a function in this file
    :return: the function
    """
    if name == 'CDM':
        return CDM
    elif name == 'WDM':
        return WDM
    elif name == 'SIDM_core_collapse':
        return SIDM_core_collapse
    elif name == 'ULDM':
        return ULDM
    elif name == 'CDMEmulator':
        return CDMFromEmulator
    elif name == 'WDM_mixed':
        return WDM_mixed
    elif name == 'WDMGeneral':
        return WDMGeneral
    elif name == "DMGalacticus":
        return DMFromGalacticus
    else:
        raise Exception('preset model '+ str(name)+' not recognized!')
        



        


    



    





        

