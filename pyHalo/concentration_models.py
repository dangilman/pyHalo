from pyHalo.Halos.concentration import *
from copy import deepcopy
import numpy as np

def preset_concentration_models(model_name, kwargs_model=None):
    """
    This function loads the concentration models, and implements some specific models from the literature
    :param model_name: the name of the concentration model (a string)
    :param kwargs_model: keyword arguments for the model class
    :return: the concentration model
    """
    if kwargs_model is None:
        kwargs_model_return = {}
    else:
        kwargs_model_return = deepcopy(kwargs_model)

    if model_name == 'DIEMERJOYCE19':
        return ConcentrationDiemerJoyce, kwargs_model_return
    elif model_name == 'PEAK_HEIGHT_POWERLAW':
        return ConcentrationPeakHeight, kwargs_model_return
    elif model_name == 'WDM_HYPERBOLIC':
        return ConcentrationWDMHyperbolic, kwargs_model_return
    elif model_name == 'WDM_POLYNOMIAL':
        return ConcentrationWDMPolynomial, kwargs_model_return
    elif model_name == 'BOSE2016':
        # https://ui.adsabs.harvard.edu/abs/2016MNRAS.455..318B/abstract
        kwargs_model_return['concentration_cdm_class'] = ConcentrationDiemerJoyce
        kwargs_model_return['kwargs_cdm'] = {}
        kwargs_model_return['c_scale'] = 60.0
        kwargs_model_return['c_power'] = -0.17
        kwargs_model_return['c_power_inner'] = 1.0
        kwargs_model_return['mc_suppression_redshift_evolution'] = True
        return ConcentrationWDMPolynomial, kwargs_model_return
    elif model_name == 'LAROCHE2022':
        # https://ui.adsabs.harvard.edu/abs/2022MNRAS.517.1867L/abstract
        # calibrated for ultra-light dark matter
        kwargs_model_return['concentration_cdm_class'] = ConcentrationDiemerJoyce
        kwargs_model_return['kwargs_cdm'] = {}
        kwargs_model_return['c_scale'] = 21.42
        kwargs_model_return['c_power'] = -0.42
        kwargs_model_return['c_power_inner'] = 1.62
        return ConcentrationWDMPolynomial, kwargs_model_return
    elif model_name == 'FROM_FORMATION_HISTORY':
        norm, slope = 0.75, 0.4
        a = norm * (0.7 / kwargs_model['dlogT_dlogk']) ** slope
        norm, slope, shift = 0.94, 0.7, 0.6
        b = norm * abs(np.log(1.76 + shift) / np.log(kwargs_model['dlogT_dlogk'] + shift)) ** slope
        kwargs_model_return['a'] = a
        kwargs_model_return['b'] = b
        del kwargs_model_return['dlogT_dlogk']
        return ConcentrationWDMHyperbolic, kwargs_model_return
    elif model_name == 'CUSTOM':
        if not hasattr(kwargs_model_return['custom_class'], 'nfw_concentration'):
            raise Exception('a custom concentration-mass relation class must have a method nfw_concentration that'
                            'takes as input (halo mass, redshift) and returns the concentration')
        custom_mc_relation_class = kwargs_model_return['custom_class']
        del kwargs_model_return['custom_class']
        return custom_mc_relation_class, kwargs_model_return
    else:
        raise Exception('model name '+str(model_name)+' not recognized')
