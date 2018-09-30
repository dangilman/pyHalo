from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Cosmology.lens_cosmo import LensCosmo
from pyHalo.Cosmology.geometry import Geometry
from pyHalo.Cosmology.lensing_mass_function import LensingMassFunction
import numpy as np
from pyHalo.Massfunc.los import LOSPowerLaw
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

        for n in range(nrealizations):
            realizations.append(self._render_single(type, args))

        return realizations

    def _LOS_mass_func(self, args):

        if not hasattr(self, 'halo_mass_function'):

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

        executables, mass_def = self._build(type, args)

        mdefs = []
        mdef_args = []

        for i, (mdef, func) in enumerate(zip(mass_def, executables)):

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

                newargs = self._mdef_args(mdef, m[j], r3[j], z[j], args)

                mdef_args.append(newargs)

        if not hasattr(self, '_geometry'):
            self._geometry = Geometry(self._cosmology,self.zlens,self.zsource,None,args['cone_opening_angle'])

        realization = Realization(masses, xpos, ypos, r2d, r3d, mdefs, redshifts, mdef_args, self._geometry)

        return realization

    def _build_los(self, args):

        mfunc_LOS = self._LOS_mass_func(args)

        los = LOSPowerLaw(args, mfunc_LOS, self.zlens, self._geometry._min_delta_z)
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

        if model_name == 'composite_powerlaw':

            mdef_los, los = self._build_los(model_args)
            mdef_main, main = self._build_main(model_args)

            executables = [los, main]
            mdefs = [mdef_los, mdef_main]

        elif model_name == 'main_lens':

            mdef_main, main = self._build_main(model_args)

            executables = [main]
            mdefs = [mdef_main]

        elif model_name == 'line_of_sight':

            mdef_los, los = self._build_los(model_args)

            executables = [los]
            mdefs = [mdef_los]

        else:
            raise ValueError('model name '+str(model_name)+' not recognized.')

        return executables, mdefs

    def _mdef_args(self, mdef, masses, r3d, redshifts, args):

        mdef_args = {}

        if mdef == 'NFW' or mdef == 'TNFW':

            nfw_c = self._lens_cosmo.NFW_concentration(masses, redshifts, logmhm=args['log_m_break'],
                                                g1=args['c_scale'],g2=args['c_power'])
            mdef_args.update({'concentration':nfw_c})

        if mdef == 'TNFW':

            truncation = self._lens_cosmo.NFW_truncation(masses, nfw_c, r3d, redshifts, self.zlens)
            mdef_args.update({'r_trunc':truncation})

        return mdef_args

if False:

    h = pyHalo(0.5,1.5)

    halo_args = {'mdef_main':'TNFW','mdef_los':'TNFW','fsub':0.01,'log_mlow':6,'log_mhigh':10, 'power_law_index': -1.9, 'log_m_break':0,
                               'parent_m200': 10**13, 'parent_c':3,'mdef':'TNFW','break_index':-1.3,'c_scale':60,
                                    'c_power':-0.17, 'r_tidal':'0.4Rs', 'break_index':-1.3,'c_scale':60,'c_power':-0.17,
                            'cone_opening_angle':6}

    real = h.render('composite_powerlaw',halo_args)
    print(len(real[0].x))
    newreal = real[0].filter(np.array([30.72,20.4]), np.array([40.3, -0.6]), logmasscut_front=11, logmasscut_back=11, back_scale_z=0)
    print(len(newreal.x))
    #print(newreal.x)
    #print(len(real[0].x))







