from pyHalo.Halos.tidal_truncation import TruncationRN, TruncationRoche, AdiabaticTidesTruncation, TruncationSplashBack


def truncation_models(model_name):
    """
    Loads and returns methods to set the tidal truncation radius (or equivalent parameter) of halos
    """
    kwargs_model = {}
    if model_name == 'TRUNCATION_R50':
        kwargs_model['LOS_truncation_factor'] = 50
        # truncates NFW halos at r50 (for field halos, this is comparable to the splashback radius)
        return TruncationRN, kwargs_model
    elif model_name == 'TRUNCATION_RN':
        return TruncationRN, kwargs_model
    elif model_name == 'TRUNCATION_ROCHE':
        return TruncationRoche, kwargs_model
    elif model_name == 'TRUNCATION_ROCHE_GILMAN2020':
        kwargs_model['RocheNorm'] = 1.4
        kwargs_model['m_power'] = 1./3
        kwargs_model['RocheNu'] = 2./3
        return TruncationRoche, kwargs_model
    elif model_name == 'ADIABATIC_TIDES':
        return AdiabaticTidesTruncation, kwargs_model
    elif model_name == 'SPLASHBACK':
        return TruncationSplashBack, kwargs_model
    else:
        raise Exception('model '+str(model_name)+' not recognized')

