from pyHalo.Massfunc.parameterizations import *
from pyHalo.Scattering.sidm_interp import logrho
from pyHalo.Halos.HaloModels.collisionless_nfw import \
    TNFWFieldHalo, TNFWMainSubhalo, NFWFieldHalo, NFWMainSubhalo
from pyHalo.Halos.HaloModels.SIDM_nfw import truncatedSIDMMainSubhalo, truncatedSIDMFieldHalo
from pyHalo.Halos.HaloModels.base import PointMassBase

class Halo(object):

    _recognized_mass_definitions = ['NFW', 'TNFW', 'SIDM_TNFW', 'PT_MASS']

    def __init__(self, mass=None, x=None, y=None, r2d=None, r3d=None, mdef=None, z=None,
                 sub_flag = None, cosmo_m_prof=None, args={}):

        self.cosmo_prof = cosmo_m_prof

        self.mass = mass

        # x and y in arcsec
        self.x = x
        self.y = y

        # r2d and r3d in kpc
        self.r2d = r2d
        self.r3d = r3d
        if args['truncate_at_pericenter']:
            self.pericenter = self.cosmo_prof.pericenter_given_r3d(r3d)
        else:
            self.pericenter = r3d

        self.mdef = mdef
        self.z = z
        self._is_main_subhalo = sub_flag
        self._args = args
        self._unique_tag = np.random.rand()

        assert mdef in self._recognized_mass_definitions, 'mass definition '+str(mdef)+' not recognized.'

    def get_z_infall(self):

        if not hasattr(self, 'z_infall'):

            self.z_infall = self.cosmo_prof.z_accreted_from_zlens(self.mass, self.z)

        return self.z_infall

    @property
    def halo_age(self):

        if not hasattr(self, '_halo_age'):
            if 'halo_age' in self._args.keys():
                self._halo_age = self._args['halo_age']
            else:
                self._halo_age = self.cosmo_prof.lens_cosmo.astropy_cosmo.halo_age(self.z)
        return self._halo_age

    @property
    def physical_args(self):

        return self._halo_type.physical_args

    @property
    def profile_args(self):

        if not hasattr(self, '_mass_def_arg'):
            self._mass_def_arg = self._halo_type.halo_parameters

        return self._mass_def_arg

    @property
    def is_subhalo(self):
        return False

    @property
    def _halo_type(self):

        if not hasattr(self, '_halo_profile_instance'):

            if self._is_main_subhalo is True:

                if self.mdef == 'NFW':

                    halo_type = NFWMainSubhalo(self)

                elif self.mdef == 'TNFW':

                    halo_type = TNFWMainSubhalo(self)

                elif self.mdef == 'SIDM_TNFW':

                    halo_type = truncatedSIDMMainSubhalo(self)

                elif self.mdef == 'PT_MASS':

                    halo_type = PointMassBase()

            else:

                if self.mdef == 'NFW':

                    halo_type = NFWFieldHalo(self)

                elif self.mdef == 'TNFW':

                    halo_type = TNFWFieldHalo(self)

                elif self.mdef == 'SIDM_TNFW':

                    halo_type = truncatedSIDMFieldHalo(self)

                elif self.mdef == 'PT_MASS':

                    halo_type = PointMassBase()

            self._halo_profile_instance = halo_type

        return self._halo_profile_instance






