from abc import ABC, abstractmethod

class RenderingClassBase(ABC):

    """
    This is the base class for a rendering_class, or a python object that generates the masses and positions of structure
    in strong lens systems. A rendering_class in pyHalo is defined as an object that contains the following methods:

    1) render: returns the object masses [numpy array], x coordinate in arcsec [numpy array],
    y coordinate in arcsec [numpy array], three dimensional position inside host halo in physical kpc
    if the object is a subhalo, else None [numpy array], the redshifts of each object [numpy array],
    and a list of bools that specify whether the object is a subhalo, or an object in the field [python list]

    2) convergence_sheet_correction: returns the lenstronomy keywords, the lenstronomy profile name, and the redshifts of
    lens models that subtract mass from the lensing volume to correct for the mass added in the form of dark matter
    halos.

    3) keyword_parse_render: extracts just the keyword arguments required to render halos from a giant dictionary that
    specifies all keyword arguments for the mass function, spatial distribution, mass definition, etc.

    4) keys_convergence_sheets: extracts just the keyword arguments required to specify the form of the convergence
    sheet correction

    """

    @staticmethod
    def _redshift_dependent_normalization(z, normalization):
        """
        Evaluates a possibly redshift-dependent line-of-sight mass function normalization
        :param z: redshift
        :param normalization: the normalization passed to create the realization
        :return:
        """
        if callable(normalization):
            norm = normalization(z)
        else:
            norm = normalization
        return norm

    @staticmethod
    def _redshift_dependent_mass_range(z, log_mlow_object, log_mhigh_object):
        """
        Evaluates a possibly redshift-dependent minimum/maximum halo mass
        :param z: redshift
        :param log_mlow_object: either a number representing the minimum halo mass, or a callable function that returns
        log10(M_min) as a function z
        :param log_mhigh_object: either a number representing the maximum halo mass, or a callable function that returns
        log10(M_max) as a function z
        :return: the minimum and maximum halo mass (in log10)
        """

        if callable(log_mlow_object):
            log_mlow = log_mlow_object(z)
        else:
            log_mlow = log_mlow_object

        if callable(log_mhigh_object):
            log_mhigh = log_mhigh_object(z)
        else:
            log_mhigh = log_mhigh_object

        return log_mlow, log_mhigh

    @abstractmethod
    def render(self, *args, **kwargs):
        ...

    @abstractmethod
    def convergence_sheet_correction(self, *args, **kwargs):
        ...

    @staticmethod
    @abstractmethod
    def keys_convergence_sheets(keywords_master):
        ...

    @staticmethod
    @abstractmethod
    def keyword_parse_render(keywords_master):
        ...


class Rendering(RenderingClassBase):

    def __init__(self, keywords_master):

        self._keywords_master = keywords_master
        self._mass_function_model_util, self._kwargs_mass_function_model = self._setup_mass_function(keywords_master)
        super(Rendering, self).__init__()

    @staticmethod
    def _setup_mass_function(keywords_master):

        if 'log_mc' in keywords_master.keys() and keywords_master['log_mc'] is not None:

            if keywords_master['mass_function_turnover_model'] == 'SCALE_FREE':
                raise Exception('you specified a scale-free mass function (mass_function_turnover_model = SCALE_FREE) but'
                                ' also specified a break in the mass function with the log_mc keyword. This does not make sense')

            if keywords_master['mass_function_turnover_model'] == 'POLYNOMIAL':
                from pyHalo.Rendering.MassFunctions.models import PolynomialSuppression
                mass_function_model_util = PolynomialSuppression()
                kwargs_mass_function_model = {}
                for kw in ['log_mc', 'a_wdm', 'b_wdm', 'c_wdm']:
                    kwargs_mass_function_model.update({kw: keywords_master[kw]})

            elif keywords_master['mass_function_turnover_model'] == 'MIXED_DM':

                from pyHalo.Rendering.MassFunctions.models import MixedDMSuppression
                mass_function_model_util = MixedDMSuppression()
                kwargs_mass_function_model = {}
                for kw in ['log_mc', 'a_wdm', 'b_wdm', 'c_wdm', 'mixed_DM_frac']:
                    kwargs_mass_function_model.update({kw: keywords_master[kw]})

            else:
                raise Exception('mass_function_turnover_model = '+str(keywords_master['mass_function_turnover_model']) +
                                'not recognized. Must be either POLYNOMIAL or MIXED_DM')
        else:
            from pyHalo.Rendering.MassFunctions.models import ScaleFree
            mass_function_model_util = ScaleFree()
            kwargs_mass_function_model = {}

        return mass_function_model_util, kwargs_mass_function_model

