from pyHalo.Halos.concentration import *

def preset_concentration_models(model_name):
    """
    This function loads the concentration models, and implements some specific models from the literature
    :param model_name: the name of the concentration model (a string)
    :param kwargs_model: keyword arguments for the model class
    :return: the concentration model
    """
    kwargs_model = {}
    if model_name == 'DIEMERJOYCE19':
        return ConcentrationDiemerJoyce, kwargs_model
    elif model_name == 'PEAK_HEIGHT_POWERLAW':
        return ConcentrationPeakHeight, kwargs_model
    elif model_name == 'WDM_HYPERBOLIC':
        return ConcentrationWDMHyperbolic, kwargs_model
    elif model_name == 'WDM_POLYNOMIAL':
        return ConcentrationWDMPolynomial, kwargs_model
    elif model_name == 'BOSE2016':
        # https://ui.adsabs.harvard.edu/abs/2016MNRAS.455..318B/abstract
        kwargs_model['concentration_cdm_class'] = ConcentrationDiemerJoyce
        kwargs_model['kwargs_cdm'] = {}
        kwargs_model['c_scale'] = 60.0
        kwargs_model['c_power'] = -0.17
        kwargs_model['c_power_inner'] = 1.0
        kwargs_model['mc_suppression_redshift_evolution'] = True
        return ConcentrationWDMPolynomial, kwargs_model
    elif model_name == 'LAROCHE2022':
        # https://ui.adsabs.harvard.edu/abs/2022MNRAS.517.1867L/abstract
        # calibrated for ultra-light dark matter
        kwargs_model['concentration_cdm_class'] = ConcentrationDiemerJoyce
        kwargs_model['kwargs_cdm'] = {}
        kwargs_model['c_scale'] = 21.42
        kwargs_model['c_power'] = -0.42
        kwargs_model['c_power_inner'] = 1.62
        return ConcentrationWDMPolynomial, kwargs_model
    else:
        raise Exception('model name '+str(model_name)+' not recognized')


