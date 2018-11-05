from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Cosmology.lens_cosmo import LensCosmo
from pyHalo.Cosmology.geometry import Geometry
from pyHalo.Cosmology.lensing_mass_function import LensingMassFunction
import numpy as np
from pyHalo.Massfunc.los import LOSPowerLaw, LOSDelta
from pyHalo.Massfunc.mainlens import MainLensPowerLaw
from pyHalo.defaults import *
from pyHalo.single_realization import Realization

class pyHalo(object):

    def __init__(self, zlens, zsource, cosmo_args = {},
                 halo_mass_function_args=None, kwargs_massfunc = {}):

        self.zlens = zlens
        self.zsource = zsource

        self._cosmology = Cosmology(**cosmo_args)
        self._lens_cosmo = LensCosmo(zlens, zsource)

        if halo_mass_function_args is None:
            halo_mass_function_args = {'model':default_mass_function, 'mdef': default_mdef}
        self._halo_mass_function_args = halo_mass_function_args
        self._kwargs_massfunc = kwargs_massfunc

    def render(self, type, args, nrealizations=1):

        realizations = []

        if not isinstance(type, list):
            type = [type]
        if not isinstance(args, list):
            args = [args]

        for n in range(nrealizations):
            realizations.append(self._render_single(type, args))

        return realizations

    def _LOS_mass_func(self, args):

        if not hasattr(self, 'halo_mass_function'):

            if 'mass_func_type' not in args.keys():
                args['mass_func_type'] = 'composite_powerlaw'

            if args['mass_func_type'] == 'delta':

                logLOS_mlow = args['M'] - 0.01
                logLOS_mhigh = args['M'] + 0.01

            else:
                if 'log_mlow_los' not in args.keys():
                    assert 'log_mlow' in args.keys()
                    logLOS_mlow = args['log_mlow']
                else:
                    logLOS_mlow = args['log_mlow_los']
                if 'log_mhigh_los' not in args.keys():
                    assert 'log_mhigh' in args.keys()
                    logLOS_mhigh = args['log_mhigh']
                else:
                    logLOS_mhigh = args['log_mhigh_los']

            assert 'cone_opening_angle' in args.keys()

            self.halo_mass_function = LensingMassFunction(self._cosmology, 10 ** logLOS_mlow, 10 ** logLOS_mhigh, self.zlens, self.zsource,
                                                          cone_opening_angle=args['cone_opening_angle'],
                                                          model_kwargs=self._halo_mass_function_args,
                                                          **self._kwargs_massfunc)

            self._geometry = self.halo_mass_function.geometry

        return self.halo_mass_function

    def _render_single(self, type, args):

        executables_list, mass_def_list, model_names = self._build(type, args)

        mdefs = []
        mdef_args = []

        init = True
        for component_index, (executables, mass_def) in enumerate(zip(executables_list, mass_def_list)):

            for i, (mdef, func) in enumerate(zip(mass_def, executables)):

                model_name = model_names[component_index][i]

                m, x, y, r2, r3, z = func()
                L = int(len(m))
                mdefs += [mdef] * L

                if i == 0:
                    masses, xpos, ypos, r2d, r3d, redshifts = m, x, y, r2, r3, z

                else:

                    masses = np.append(masses, m)
                    xpos = np.append(xpos, x)
                    ypos = np.append(ypos, y)
                    r2d = np.append(r2d, r2)
                    r3d = np.append(r3d, r3)
                    redshifts = np.append(redshifts, z)

                for j in range(L):

                    newargs = self._mdef_args(mdef, m[j], r3[j], z[j], args[component_index], model_name)

                    mdef_args.append(newargs)

            if not hasattr(self, '_geometry'):
                self._geometry = Geometry(self._cosmology, self.zlens, self.zsource, args[0]['cone_opening_angle'])

            if 'log_m_break' in args[component_index].keys() and 'break_index' in args[component_index].keys():
                wdm_params = {'log_m_break': args[0]['log_m_break'], 'break_index': args[0]['break_index']}
                if 'LOS_normalization' in args[0].keys():
                    wdm_params.update({'LOS_normalization': args[0]['LOS_normalization']})

            else:
                wdm_params = None

            mass_sheet = True
            if 'mass_func_type' in args[component_index].keys():
                if args[component_index]['mass_func_type'] == 'delta':
                    mass_sheet = False

            if init:
                realization = Realization(masses, xpos, ypos, r2d, r3d, mdefs, redshifts, mdef_args, self.halo_mass_function,
                                      wdm_params=wdm_params, mass_sheet_correction=mass_sheet)
                init = False
            else:
                new = Realization(masses, xpos, ypos, r2d, r3d, mdefs, redshifts, mdef_args, self.halo_mass_function,
                                      wdm_params=wdm_params, mass_sheet_correction=mass_sheet)
                realization = realization.join(new)

        return realization

    def _build_los(self, args):

        mfunc_LOS = self._LOS_mass_func(args)

        if 'mass_func_type' in args and args['mass_func_type'] == 'delta':
            los = LOSDelta(args, mfunc_LOS)
        else:
            # default to a power law
            los = LOSPowerLaw(args, mfunc_LOS)

        if 'mdef_los' not in args.keys():
            raise ValueError('specify mass definition for line of sight halos with mdef_los.')
        mdef = args['mdef_los']

        return mdef, los

    def _build_main(self, args):

        mdef = args['mdef_main']

        if 'mdef_main' not in args.keys():
            raise ValueError('specify mass definition for lens plane halos with mdef_main.')

        return mdef, MainLensPowerLaw(args, self._lens_cosmo)

    def _build(self, model_name, model_args):

        executables = []
        mdefs = []
        mod_name = []

        for mod, args in zip(model_name, model_args):

            _ = self._LOS_mass_func(args)

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

    def _mdef_args(self, mdef, masses, r3d, redshifts, args, model_name):

        mdef_args = {}

        if mdef == 'NFW' or mdef == 'TNFW':

            nfw_c = self._lens_cosmo.NFW_concentration(masses, redshifts, logmhm=args['log_m_break'],
                                                g1=args['c_scale'],g2=args['c_power'])
            mdef_args.update({'concentration':nfw_c})

        if mdef == 'TNFW':

            if model_name == 'LOS':
                truncation = self._lens_cosmo.NFW_truncation(masses, redshifts)
            else:
                truncation = self._lens_cosmo.truncation_roche(masses, r3d)

            mdef_args.update({'r_trunc':truncation})

        if mdef == 'PJAFFE':

            truncation = self._lens_cosmo.truncation_roche(masses, r3d)
            mdef_args.update({'r_trunc':truncation})
        if mdef == 'POINT_MASS':

            pass

        return mdef_args







