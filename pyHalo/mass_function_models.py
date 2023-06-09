from pyHalo.Rendering.MassFunctions.mass_function_base import *
from pyHalo.Rendering.MassFunctions.density_peaks import *
from copy import deepcopy


def preset_mass_function_models(model_name, kwargs_model={}):
    """
    This function loads the concentration models, and implements some specific models from the literature
    :param model_name: the name of the concentration model (a string)
    :param kwargs_model: keyword arguments for the model class
    :return: the concentration model
    """
    kwargs_model_out = deepcopy(kwargs_model)
    if model_name == 'SHMF_LOVELL2020':
        # calibrated for subhalos
        # from https://arxiv.org/pdf/2003.01125.pdf
        kwargs_model_out['a_wdm'] = 4.2
        kwargs_model_out['b_wdm'] = 2.5
        kwargs_model_out['c_wdm'] = -0.2
        return WDMPowerLaw, kwargs_model_out
    elif model_name == 'LOVELL2020':
        # calibrated for field halos
        # from https://arxiv.org/pdf/2003.01125.pdf
        kwargs_model_out['a_wdm'] = 2.3
        kwargs_model_out['b_wdm'] = 0.8
        kwargs_model_out['c_wdm'] = -1.0
        return ShethTormenTurnover, kwargs_model_out
    elif model_name == 'SHMF_LOVELL2014':
        kwargs_model_out['a_wdm'] = 1.0
        kwargs_model_out['b_wdm'] = 1.0
        kwargs_model_out['c_wdm'] = -1.3
        return WDMPowerLaw, kwargs_model_out
    elif model_name == 'LOVELL2014':
        kwargs_model_out['a_wdm'] = 1.0
        kwargs_model_out['b_wdm'] = 1.0
        kwargs_model_out['c_wdm'] = -1.3
        return ShethTormenTurnover, kwargs_model_out
    elif model_name == 'SCHIVE2016':
        # https://arxiv.org/pdf/1508.04621.pdf
        # calibrated for ultra-light dark matter
        kwargs_model_out['a_wdm'] = 1.0
        kwargs_model_out['b_wdm'] = 1.1
        kwargs_model_out['c_wdm'] = -2.2
        return ShethTormenTurnover, kwargs_model_out
    elif model_name == 'SHMF_SCHIVE2016':
        # https://arxiv.org/pdf/1508.04621.pdf
        # calibrated for ultra-light dark matter
        kwargs_model_out['a_wdm'] = 1.0
        kwargs_model_out['b_wdm'] = 1.1
        kwargs_model_out['c_wdm'] = -2.2
        return WDMPowerLaw, kwargs_model_out
    elif model_name == 'POWER_LAW':
        return CDMPowerLaw, kwargs_model_out
    elif model_name == 'POWER_LAW_TURNOVER':
        return WDMPowerLaw, kwargs_model_out
    elif model_name == 'SHMF_MIXED_WDM_TURNOVER':
        return MixedWDMPowerLaw, kwargs_model_out
    elif model_name == 'MIXED_WDM_TURNOVER':
        return ShethTormenMixedWDM, kwargs_model_out
    elif model_name == 'STUCKER':
        if 'dlogT_dlogk' not in kwargs_model.keys():
            raise Exception('Must specify |dlogT_dlogk| (absolute value of the ' \
            'logarithmic derivative of the transfer function) when using the STUCKER model.)')
        if 'a_wdm' in kwargs_model.keys():
            raise Exception('Cannot specify a_wdm with the Stucker model.')
        if 'b_wdm' in kwargs_model.keys():
            raise Exception('Cannot specify b_wdm with the Stucker model.')
        if 'c_wdm' in kwargs_model.keys():
            raise Exception('Cannot specify c_wdm with the Stucker model.')
        from pyHalo.Rendering.MassFunctions.stucker import stucker_suppression_params
        a_wdm, b_wdm, c_wdm = stucker_suppression_params(kwargs_model_out['dlogT_dlogk'])
        kwargs_model_out['a_wdm'] = a_wdm
        kwargs_model_out['b_wdm'] = b_wdm
        kwargs_model_out['c_wdm'] = c_wdm
        del kwargs_model_out['dlogT_dlogk']
        return WDMPowerLaw, kwargs_model_out
    else:
        raise Exception('model name '+str(model_name)+' not recognized')


