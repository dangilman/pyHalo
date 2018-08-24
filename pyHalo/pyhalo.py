from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Cosmology.lens_cosmo import LensCosmo
from pyHalo.Cosmology.lensing_mass_function import LensingMassFunction
import numpy as np
from pyHalo.models import LOSPowerLaw, MainLensPowerLaw
from pyHalo.defaults import *
from pyHalo.single_realization import Realization

"""Main module."""

class pyHalo(object):

    def __init__(self, zlens, zsource, cone_opening_angle, cosmo_args = {},
                 halo_mass_function_args={'model':'reed07'}, multiplane=True,
                 logLOS_mlow = None, logLOS_mhigh = None):

        self.zlens = zlens
        self.zsource = zsource

        self._multiplane = multiplane
        self._logLOS_mlow, self._logLOS_mhigh = logLOS_mlow, logLOS_mhigh

        self._cosmology = Cosmology(**cosmo_args)
        self._lens_cosmo = LensCosmo(zlens, zsource)
        self._cone_opening = cone_opening_angle

        if self._multiplane:

            self.halo_mass_function = LensingMassFunction(self._cosmology, logLOS_mlow, logLOS_mhigh, zlens, zsource,
                                                          cone_opening_angle)

    def _build(self, halo_args, n_mass_components):

        mass_definitions = []
        prescription = []

        for a in range(0,n_mass_components):

            mdef = []
            pres = []

            if self._multiplane:

                args_main_spatial, args_main_massfunc = self._split_args(halo_args[a], 'main')
                _, args_LOS_massfunc = self._split_args(halo_args[a], 'LOS')

                main = MainLensPowerLaw(args_main_massfunc, args_main_spatial)
                los = LOSPowerLaw(self.halo_mass_function, args_LOS_massfunc, self.zlens, default_zstep)
                pres += [los, main]
                mdef += [halo_args[a]['mdef']]*2

            else:

                args_main_spatial, args_main_massfunc = self._split_args(halo_args[a], 'main')
                main = MainLensPowerLaw(args_main_massfunc, args_main_spatial)
                pres += [main]
                mdef += [halo_args[a]['mdef']]

            mass_definitions.append(mdef)
            prescription.append(pres)

            for key in args_main_massfunc.keys():
                if key not in halo_args[a].keys():
                    halo_args[a][key] = args_main_massfunc[key]

        return prescription, mass_definitions, halo_args

    def _mdef_args(self, masses, r3d, mdef, redshifts, args):

        mdef_args = []

        if mdef == 'NFW' or mdef == 'TNFW':
            print(redshifts)
            nfw_c = self._lens_cosmo.NFW_concentration(masses, redshifts, logmhm=0,
                                                g1=args['c_scale'],g2=args['c_power'])
            mdef_args += [{'concentration':nfw_c}]

        if mdef == 'TNFW':

            truncation = self._lens_cosmo.nfw_truncation(masses, redshifts, nfw_c, r3d)
            mdef_args += [{'r_trunc':truncation}]

        if mdef == 'pointmass':

            cosmo_factor = self._lens_cosmo.point_mass_fac(masses, redshifts)
            mdef_args += [{'cosmo_factor':cosmo_factor}]

        return mdef_args

    def render(self, nrealizations=1, halo_args = [], n_mass_components=1):

        realizations = []

        for n in range(0,nrealizations):

            executables, mass_def, halo_args = self._build(halo_args, n_mass_components)

            masses = None
            mass_def_args = []

            for k in range(0, n_mass_components):

                m_args_new = []

                for i,func in enumerate(executables[n]):

                    if masses is None:
                        masses, xpositions, ypositions, r2d, r3d, redshift = func()
                        redshift[np.where(redshift == None)] = self.zlens
                        mdefs = [mass_def[k][i]]*len(masses)


                    else:

                        m, x, y, r2, r3, z = func()
                        z[np.where(z == None)] = self.zlens

                        masses = np.append(masses,m)
                        xpositions = np.append(xpositions,x)
                        ypositions = np.append(ypositions,y)
                        r2d = np.append(r2d, r2)
                        r3d = np.append(r3d, r3)
                        redshift = np.append(redshift,z)

                        mdefs += [mass_def[k][i]*len(masses)]

                    m_args_new += self._mdef_args(masses, r3d, mass_def[k][i], redshift, halo_args[k])

                mass_def_args.append(m_args_new)

            realizations.append(Realization(masses, xpositions, ypositions, r2d, r3d, mdefs, redshift, mass_def_args))

        return realizations

    def _split_args(self, args, env):

        new_args_spatial = {}
        new_args_spatial['rmax2d'] = self._cone_opening * 0.5

        if env == 'main':

            if 'parent_m200' in args and 'parent_c' in args.keys():
                rho0_kpc, parent_Rs, parent_r200 = self._lens_cosmo.NFW_params(args['parent_m200'],args['parent_c'],self.zlens)
                new_args_spatial['Rs'] = parent_Rs
                new_args_spatial['rmax3d'] = parent_r200
            else:
                try:
                    new_args_spatial['Rs'] = args['parent_Rs']
                    new_args_spatial['rmax3d'] = args['parent_r200']
                except:
                    raise ValueError('must specify either (parent_c, m200) for parent halo, or '
                                     '(parent_Rs, parent_r200) directly')

            if 'r_tidal_parent' in args.keys():
                if isinstance(args['r_tidal_parent'],str):
                    if args['r_tidal_parent'] == 'Rs':
                        new_args_spatial['r_core'] = new_args_spatial['Rs']
                    else:
                        if args['r_tidal_parent'][-2:] != 'Rs':
                            raise ValueError('if specifying the tidal core radius as number*Rs, the last two '
                                             'letters in the string must be "Rs".')

                        scale = float(args['r_core'][:-2])
                        new_args_spatial['r_core'] = scale*new_args_spatial['Rs']

                else:
                    new_args_spatial['r_core'] = args['r_tidal_parent']

        new_args_massfunc = {}

        if env == 'main':

            required_keys = ['power_law_index', 'log_mlow', 'log_mhigh', 'log_m_break']

            for key in required_keys:
                try:
                    new_args_massfunc[key] = args[key]
                except:
                    raise ValueError('must specify a value for '+key)

            if 'norm_A0' in args.keys():

                new_args_massfunc['normalization'] = args['norm_A0']

            elif 'fsub' in args.keys():

                new_args_massfunc['normalization'] = self._lens_cosmo.convert_fsub_to_norm(args['fsub'],
                                            self._cone_opening,new_args_massfunc['power_law_index'],
                                                  10**new_args_massfunc['log_mlow'],10**new_args_massfunc['log_mhigh'])
            else:
                raise ValueError('must either specify the normalization "norm_A0" direclty, or specify the mass fraction'
                                 'in substructure at the Einstein radius "fsub".')

        elif env == 'LOS':

            required_keys = ['zmin', 'zmax','log_m_break']

            for key in required_keys:
                try:
                    new_args_massfunc[key] = args[key]
                except:
                    if key == 'zmin':
                        new_args_massfunc['zmin'] = 0
                    else:
                        new_args_massfunc['zmax'] = self.zsource
            new_args_massfunc['log_mlow'] = self._logLOS_mlow
            new_args_massfunc['log_mhigh'] = self._logLOS_mhigh

        if new_args_massfunc['log_m_break'] == 0:
            new_args_massfunc['break_index'] = 0
            new_args_massfunc['c_scale'] = 0
            new_args_massfunc['c_power'] = 0

        else:
            try:
                new_args_massfunc['break_index'] = args['break_index']
                new_args_massfunc['c_scale'] = args['c_scale']
                new_args_massfunc['c_power'] = args['c_power']
            except:
                raise ValueError('must specify a value for "break_index, c_scale, c_power" if log_m_break != 0 '
                                 '(because you are specifying a WDM scenario in which the concentration and mass function'
                                 'slope  of halos is affected')

        return new_args_spatial, new_args_massfunc

h = pyHalo(0.5,1.5,6,logLOS_mlow=9,logLOS_mhigh=10, multiplane=False)

halo_args = {'fsub':0.01,'log_mlow':9,'log_mhigh':10, 'power_law_index': -1.9, 'log_m_break':8, 'parent_m200':10**13,
             'parent_c':3,'mdef':'NFW','break_index':-1.3,'c_scale':60,'c_power':-0.03}

real = h.render(1,[halo_args])
print(real[0].x.shape)





