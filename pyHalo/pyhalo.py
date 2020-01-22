from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Cosmology.geometry import Geometry
from pyHalo.Cosmology.lensing_mass_function import LensingMassFunction
import numpy as np
from pyHalo.Massfunc.los import LOSPowerLaw, LOSDelta
from pyHalo.Massfunc.mainlens import MainLensPowerLaw
from pyHalo.defaults import *
from pyHalo.single_realization import Realization


class pyHalo(object):

    def __init__(self, zlens, zsource, cosmology_kwargs={},
                 kwargs_halo_mass_function={}):

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

    def reset_redshifts(self, zlens, zsource):

        self.zlens = zlens
        self.zsource = zsource
        self._cosmology = Cosmology(**self._cosmology_kwargs)
        self.halo_mass_function = None
        self._geometry = None

    @property
    def astropy_cosmo(self):
        return self._cosmology.astropy

    def render(self, type, args, nrealizations=1, verbose=False):

        realizations = []

        if not isinstance(type, list):
            type = [type]
        if not isinstance(args, list):
            args = self._add_profile_params(args)
            args = [args]
        else:
            for i, ai in enumerate(args):
                args[i] = self._add_profile_params(ai)

        for n in range(nrealizations):
            realizations.append(self._render_single(type, args, verbose))

        return realizations

    def _build_LOS_mass_function(self, args):

        if self.halo_mass_function is None:

            if 'mass_func_type' not in args.keys():
                args['mass_func_type'] = realization_default.default_type

            if args['mass_func_type'] == 'delta':

                logLOS_mlow = args['M'] - 0.01
                logLOS_mhigh = args['M'] + 0.01

            else:
                if 'log_mlow_los' not in args.keys():
                    logLOS_mlow = realization_default.log_mlow
                else:
                    logLOS_mlow = args['log_mlow_los']

                if 'log_mhigh_los' not in args.keys():
                    logLOS_mhigh = realization_default.log_mhigh
                else:
                    logLOS_mhigh = args['log_mhigh_los']

            if 'two_halo_term' not in self._halo_mass_function_args.keys():
                self._halo_mass_function_args.update({'two_halo_term': realization_default.two_halo_term})
            if 'mass_function_model' not in self._halo_mass_function_args.keys():
                self._halo_mass_function_args.update({'mass_function_model': cosmo_default.default_mass_function})

            self.halo_mass_function = LensingMassFunction(self._cosmology, 10 ** logLOS_mlow, 10 ** logLOS_mhigh, self.zlens, self.zsource,
                                                          cone_opening_angle=args['cone_opening_angle'],
                                                          **self._halo_mass_function_args)

        return self.halo_mass_function

    def _render_single(self, type, args, verbose):

        self.halo_mass_function = self._build_LOS_mass_function(args[0])
        self._geometry = self.halo_mass_function.geometry

        executables_list, mass_def_list, model_names = self._build(type, args)

        mdefs = []

        init = True
        for component_index, (executables, mass_def) in enumerate(zip(executables_list, mass_def_list)):

            for i, (mdef, func) in enumerate(zip(mass_def, executables)):

                if verbose:
                    if i==0:
                        print('generating first executable... ')
                    else:
                        print('generating second executable... ')

                m, x, y, r2, r3, z, sub_flag = func()

                L = int(len(m))
                mdefs += [mdef] * L

                if i == 0:
                    masses, xpos, ypos, r2d, r3d, redshifts = m, x, y, r2, r3, z
                    subhalo_flag = sub_flag

                else:

                    masses = np.append(masses, m)
                    xpos = np.append(xpos, x)
                    ypos = np.append(ypos, y)
                    r2d = np.append(r2d, r2)
                    r3d = np.append(r3d, r3)
                    redshifts = np.append(redshifts, z)
                    subhalo_flag += sub_flag

            if self._geometry is None:
                self._geometry = self._set_geometry(args[0]['cone_opening_angle'])

            mass_sheet = True
            if 'mass_func_type' in args[component_index].keys():
                if args[component_index]['mass_func_type'] == 'delta':
                    mass_sheet = False

            if init:
                if verbose:
                    print('computing halo physical parameters... ')
                    if 'mc_model' in args[component_index].keys():
                        print('using mc-relation: ', args[component_index]['mc_model'])
                realization = Realization(masses, xpos, ypos, r2d, r3d, mdefs, redshifts, subhalo_flag, self.halo_mass_function,
                                          other_params=args[component_index], mass_sheet_correction=mass_sheet)
                init = False
            else:
                if verbose:
                    print('computing halo physical parameters... ')
                    if 'mc_model' in args[component_index].keys():
                        print('using mc-relation: ', args[component_index]['mc_model'])
                new = Realization(masses, xpos, ypos, r2d, r3d, mdefs, redshifts, subhalo_flag, self.halo_mass_function,
                                  other_params=args[component_index], mass_sheet_correction=mass_sheet)
                realization = realization.join(new)

        return realization

    def _add_profile_params(self, args):

        return set_default_kwargs(args)

    def _build_los(self, args):

        if 'mass_func_type' in args and args['mass_func_type'] == 'delta':
            los = LOSDelta(args, self.halo_mass_function)
        else:
            # default to a power law
            los = LOSPowerLaw(args, self.halo_mass_function)

        if 'mdef_los' not in args.keys():
            raise ValueError('specify mass definition for line of sight halos with mdef_los.')
        mdef = args['mdef_los']

        return mdef, los

    def _build_main(self, args):

        mdef = args['mdef_main']

        if 'mdef_main' not in args.keys():
            raise ValueError('specify mass definition for lens plane halos with mdef_main.')

        return mdef, MainLensPowerLaw(args, self._geometry)

    def _build(self, model_name, model_args):

        executables = []
        mdefs = []
        mod_name = []

        for mod, args in zip(model_name, model_args):

            _ = self._build_LOS_mass_function(args)

            if mod == 'composite_powerlaw':

                mdef_los, los = self._build_los(args)
                mdef_main, main = self._build_main(args)

                executables.append([los, main])
                mdefs.append([mdef_los, mdef_main])
                mod_name.append(['LOS','main'])

            elif mod == 'main_lens':

                mdef_main, main = self._build_main(args)

                executables.append([main])
                mdefs.append([mdef_main])
                mod_name.append(['main'])

            elif mod == 'line_of_sight':

                mdef_los, los = self._build_los(args)

                executables.append([los])
                mdefs.append([mdef_los])
                mod_name.append(['LOS'])

            else:
                raise ValueError('model name '+str(mod)+' not recognized.')

        return executables, mdefs, mod_name







