from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Cosmology.lensing_mass_function import LensingMassFunction
import numpy as np
from pyHalo.defaults import *
from scipy.interpolate import interp1d

class pyHaloBase(object):

    def __init__(self, zlens, zsource, cosmology_kwargs,
                 kwargs_halo_mass_function):

        """

        :param zlens: lens redshift
        :param zsource: source redshift
        :param cosmology_kwargs:
        keyword arguments for 'Cosmology' class. See documentation in cosmology.py
        :param halo_mass_function_args:
        keyword arguments for 'LensingMassFunction' class. See documentation in lensing_mass_function.py
        :param kwargs_massfunc:
        keyword arguments
        """

        self._cosmology_kwargs = cosmology_kwargs
        self._kwargs_mass_function = kwargs_halo_mass_function
        self._halo_mass_function_args = kwargs_halo_mass_function
        self.reset_redshifts(zlens, zsource)

    def lens_plane_redshifts(self, kwargs_render={}):

        """
        This routine sets up the redshift planes along the line of sight in the lens system
        :param kwargs_render: keyword arguments, if none are specified default values will be used (see defaults.py)
        :return: lens plane redshifts and the thickness of each slice
        """

        zmin = lenscone_default.default_zstart
        if 'zstep' not in kwargs_render.keys():
            zstep = lenscone_default.default_z_step
        else:
            zstep = kwargs_render['zstep']

        front_z = np.arange(zmin, self.zlens, zstep)
        back_z = np.arange(self.zlens, self.zsource, zstep)
        redshifts = np.append(front_z, back_z)

        delta_zs = []
        for i in range(0, len(redshifts) - 1):
            delta_zs.append(redshifts[i + 1] - redshifts[i])
        delta_zs.append(self.zsource - redshifts[-1])

        return list(np.round(redshifts, 2)), np.round(delta_zs, 2)

    def reset_redshifts(self, zlens, zsource):

        self.zlens = zlens
        self.zsource = zsource
        self.cosmology = Cosmology(**self._cosmology_kwargs)
        self.halo_mass_function = None
        self.geometry = None

    @property
    def astropy_cosmo(self):
        return self.cosmology.astropy

    def build_LOS_mass_function(self, args):

        if self.halo_mass_function is None:

            if 'log_mlow_los' not in args.keys():
                logLOS_mlow = realization_default.log_mlow
            else:
                logLOS_mlow = args['log_mlow_los']

            if 'log_mhigh_los' not in args.keys():
                logLOS_mhigh = realization_default.log_mhigh
            else:
                logLOS_mhigh = args['log_mhigh_los']

            if 'mass_function_model' not in self._halo_mass_function_args.keys():
                self._halo_mass_function_args.update({'mass_function_model': cosmo_default.default_mass_function})

            self.halo_mass_function = LensingMassFunction(self.cosmology, self.zlens, self.zsource,
                                                          10 ** logLOS_mlow, 10 ** logLOS_mhigh,
                                                          cone_opening_angle=args['cone_opening_angle'],
                                                          **self._halo_mass_function_args)

        return self.halo_mass_function

