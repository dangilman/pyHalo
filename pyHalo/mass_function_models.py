from pyHalo.Rendering.MassFunctions.mass_function_base import *
from pyHalo.Rendering.MassFunctions.density_peaks import *

def preset_mass_function_models(model_name):
    """
    This function loads the concentration models, and implements some specific models from the literature
    :param model_name: the name of the concentration model (a string)
    :param kwargs_model: keyword arguments for the model class
    :return: the concentration model
    """
    kwargs_model = {}
    if model_name == 'SHMF_LOVELL2020':
        # calibrated for subhalos
        # from https://arxiv.org/pdf/2003.01125.pdf
        kwargs_model['a_wdm'] = 4.2
        kwargs_model['b_wdm'] = 2.5
        kwargs_model['c_wdm'] = -0.2
        return WDMPowerLaw, kwargs_model
    elif model_name == 'LOVELL2020':
        # calibrated for field halos
        # from https://arxiv.org/pdf/2003.01125.pdf
        kwargs_model['a_wdm'] = 2.3
        kwargs_model['b_wdm'] = 0.8
        kwargs_model['c_wdm'] = -1.0
        return ShethTormenTurnover, kwargs_model
    elif model_name == 'SHMF_LOVELL2014':
        kwargs_model['a_wdm'] = 1.0
        kwargs_model['b_wdm'] = 1.0
        kwargs_model['c_wdm'] = -1.3
        return WDMPowerLaw, kwargs_model
    elif model_name == 'LOVELL2014':
        kwargs_model['a_wdm'] = 1.0
        kwargs_model['b_wdm'] = 1.0
        kwargs_model['c_wdm'] = -1.3
        return ShethTormenTurnover, kwargs_model
    elif model_name == 'SCHIVE2016':
        # https://arxiv.org/pdf/1508.04621.pdf
        # calibrated for ultra-light dark matter
        kwargs_model['a_wdm'] = 1.0
        kwargs_model['b_wdm'] = 1.1
        kwargs_model['c_wdm'] = -2.2
        return ShethTormenTurnover, kwargs_model
    elif model_name == 'SHMF_SCHIVE2016':
        # https://arxiv.org/pdf/1508.04621.pdf
        # calibrated for ultra-light dark matter
        kwargs_model['a_wdm'] = 1.0
        kwargs_model['b_wdm'] = 1.1
        kwargs_model['c_wdm'] = -2.2
        return WDMPowerLaw, kwargs_model
    elif model_name == 'POWER_LAW':
        return CDMPowerLaw, kwargs_model
    elif model_name == 'POWER_LAW_TURNOVER':
        return WDMPowerLaw, kwargs_model
    elif model_name == 'SHMF_MIXED_WDM_TURNOVER':
        return MixedWDMPowerLaw, kwargs_model
    elif model_name == 'MIXED_WDM_TURNOVER':
        return ShethTormenMixedWDM, kwargs_model
    else:
        raise Exception('model name '+str(model_name)+' not recognized')


