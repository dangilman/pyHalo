import numpy as np
from pyHalo.Spatial.nfw import NFW_3D
from pyHalo.Massfunc.parameterizations import *

class Halo(object):

    is_subhalo = False

    has_concentration = ['NFW', 'TNFW', 'coreBURKERT', 'cBURKcNFW', 'CNFW', 'cNFWmod',
                         'cNFWmod_trunc']

    has_truncation = ['TNFW', 'cNFWmod_trunc']

    has_core = ['coreBURKERT', 'cBURKcNFW', 'CNFW', 'cNFWmod', 'cNFWmod_trunc']

    def __init__(self, mass=None, x=None, y=None, r2d=None, r3d=None, mdef=None, z=None, cosmo_m_prof=None, args={}):

        self.mass = mass

        # x and y in arcsec
        self.x = x
        self.y = y

        # r2d and r3d in kpc
        self.r2d = r2d
        self.r3d = r3d

        self.mdef = mdef
        self.z = z
        self.cosmo_prof = cosmo_m_prof
        self._args = args
        # compute these at the end
        self.mass_def_arg = self.profile_parameters()

        self._unique_tag = np.random.rand()

    def add_subhalos(self, subhalo_args):

        if self.is_subhalo is True:

            return []

        elif not hasattr(self, 'subhalos'):

            self.subhalos = []

            if self.mdef not in ['NFW', 'TNFW', 'CNFW']:
                raise Exception('subhalos only implemented for NFW-like profiles.')

            if not hasattr(self, '_spatial'):

                r200_arcsec = self.cosmo_prof.rN_M_nfw_physical_arcsec(self.mass, 200, self.z)
                c = self.cosmo_prof.NFW_concentration(self.mass, self.z, logmhm=self._args['log_m_break'],
                                                      c_scale=self._args['c_scale'], c_power=self._args['c_power'])

                rs_arcsec = r200_arcsec * c ** -1

                self._spatial = NFW_3D(rs_arcsec, r200_arcsec, r200_arcsec, xoffset=self.x, yoffset=self.y,
                                       tidal_core=False)

            if subhalo_args['mdef'] == 'POINT_MASS':
                func = self._POINT_MASS_subhalos
            elif subhalo_args['mdef'] == 'NFW':
                func = self._NFW_subhalos
            elif subhalo_args['mdef'] == 'TNFW':
                func = self._NFW_subhalos
            else:
                raise Exception('subhalo mass profile '+
                                subhalo_args['mdef']+' not recognized.')

            subhalos_exist, properties = func(subhalo_args)

            if subhalos_exist:
                msub, xsub, ysub, r2dsub, r3dsub = properties[0], properties[1], properties[2], \
                                                   properties[3], properties[4]

                # reduce the parent mass
                new_parent_mass = self.mass - np.sum(msub)

                while new_parent_mass < 0:

                    msub = np.delete(msub, np.argmax(msub))
                    new_parent_mass = self.mass - np.sum(msub)

                self.mass = new_parent_mass
                if self.mdef in self.has_concentration:
                    self.mass_def_arg['concentration'] = self.cosmo_prof.NFW_concentration(self.mass,
                                                                                           self.z, logmhm=self._args['log_m_break'],
                                                                                           c_scale=self._args['c_scale'], c_power=self._args['c_power'])

                for (mi, xi, yi, r2i, r3i) in zip(msub, xsub, ysub, r2dsub, r3dsub):
                    new_object = Halo(mass=mi, x=xi, y=yi, r2d=None, r3d=None, mdef=subhalo_args['mdef'], z=self.z,
                                      cosmo_m_prof=self.cosmo_prof, args=self._args)
                    new_object.is_subhalo = True
                    self.subhalos.append(new_object)

        return self.subhalos

    def _POINT_MASS_subhalos(self, subhalo_args):

        mfraction = subhalo_args['mass_fraction'] * self.mass * 10 ** subhalo_args['log_mean_mass'] ** -1
        nsub = np.random.poisson(mfraction)

        if nsub > 0:
            subx, suby, subr2d, subr3d = self._spatial.draw(nsub)
            mass_dis = Gaussian(subhalo_args['log_mean_mass'], 1, nsub)
            msub = mass_dis.draw()

            return True, [msub, subx, suby, subr2d, subr3d]
        else:
            return False, None

    def _NFW_subhalos(self, subhalo_args):

        if not hasattr(self, '_submfunc'):
            self._submfunc = SubhaloPowerLaw(10**subhalo_args['log_mlow'], self.mass)

        msub = self._submfunc.draw()

        if len(msub) > 0:
            subx, suby, subr2d, subr3d = self._spatial.draw(len(msub))
            return True, [msub, subx, suby, subr2d, subr3d]
        else:
            return False, None

    def profile_parameters(self):

        mdef_args = {}

        if self.mdef in self.has_concentration:

            nfw_c = self.cosmo_prof.NFW_concentration(self.mass, self.z, logmhm=self._args['log_m_break'],
                                                      c_scale=self._args['c_scale'], c_power=self._args['c_power'],
                                                      scatter=self._args['c_scatter'])
            mdef_args.update({'concentration': nfw_c})

        if self.mdef in self.has_truncation:

            if self.is_subhalo:
                truncation = self.cosmo_prof.LOS_truncation(self.mass, self.z, self._args['LOS_truncation'])
            elif self.z == self.cosmo_prof.lens_cosmo.z_lens:
                truncation = self.cosmo_prof.truncation_roche(self.mass, self.r3d, self.z, self._args['RocheNorm'],
                                                              self._args['RocheNu'])
            else:
                truncation = self.cosmo_prof.LOS_truncation(self.mass, self.z)

            mdef_args.update({'r_trunc': truncation})

        if self.mdef in self.has_core:
            mdef_args.update({'b': self._args['core_ratio']})

        if self.mdef == 'POINT_MASS':
            pass

        return mdef_args






