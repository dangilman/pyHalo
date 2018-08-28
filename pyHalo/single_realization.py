import numpy as np
from pyHalo.Lensing.NFW import NFWLensing
from pyHalo.Lensing.TNFW import TNFWLensing
from pyHalo.Lensing.PTmass import PTmasslensing

def realization_at_z(realization,z):

    m, x, y, r2, r3, mdef, redshift, mdefargs = realization.halos_at_z(z)

    return Realization(m, x, y, r2, r3, mdef, redshift, mdefargs, realization.geometry)

class Realization(object):

    def __init__(self, masses, x, y, r2d, r3d, mdefs, z, mass_def_args, geometry):

        self.masses = masses
        self.x = x
        self.y = y
        self.mdefs = mdefs
        self.redshifts = z
        self.mass_def_args = mass_def_args

        self.r2d = r2d
        self.r3d = r3d

        self.geometry = geometry
        self.lens_cosmo = geometry._lens_cosmo
        self._unique_redshifts = np.unique(self.redshifts)

        self._lensing_functions = self._lensing_list()

    def _lensing_list(self):

        lensing = []

        for i, mdef in enumerate(self.mdefs):

            if mdef == 'NFW':
                lensing.append(NFWLensing(self.lens_cosmo))

            elif mdef == 'TNFW':
                lensing.append(TNFWLensing(self.lens_cosmo))

            elif mdef == 'POINT_MASS':
                lensing.append(PTmasslensing(self.lens_cosmo))

            else:
                raise ValueError('halo profile ' + str(mdef) + ' not recongnized.')

        return lensing

    def lensing_quantities(self, mass_sheet_correction = True):

        kwargs_lens = []
        lens_model_names = []

        for i,mdef in enumerate(self.mdefs):

            args = {'x': self.x[i], 'y': self.y[i], 'mass': self.masses[i]}
            lens_model_names.append(mdef)

            if mdef == 'NFW':
                args.update({'concentration':self.mass_def_args[i]['concentration'],'redshift':self.redshifts[i]})
            elif mdef == 'TNFW':
                args.update({'concentration': self.mass_def_args[i]['concentration'], 'redshift': self.redshifts[i]})
                args.update({'r_trunc': self.mass_def_args[i]['r_trunc']})
            elif mdef == 'POINT_MASS':
                args.update({'redshift': self.redshifts[i]})
            else:
                raise ValueError('halo profile '+str(mdef)+' not recongnized.')

            kwargs_lens.append(self._lensing_functions[i].params(**args))

        if mass_sheet_correction:

            kwargs_mass_sheets, z_sheets = self.mass_sheet_correction()
            kwargs_lens += kwargs_mass_sheets
            lens_model_names += ['CONVERGENCE'] * len(kwargs_mass_sheets)
            redshift_list = np.append(self.redshifts, z_sheets)

        else:
            redshift_list = self.redshifts

        return lens_model_names, redshift_list, kwargs_lens

    def mass_sheet_correction(self):

        kwargs = []
        zsheet = []

        for z in self._unique_redshifts:

            if z != self.geometry._zlens:

                kappa = self.convergence_at_z(z)
                kwargs.append({'kappa_ext': - kappa})
                zsheet.append(z)

        return kwargs, zsheet

    def halos_at_z(self,z):

        masses, x, y, r2d, r3d, mdefs, massdefargs = [], [], [], [], [], [], []

        for i, mdef in enumerate(self.mdefs):
            if self.redshifts[i] != z:
                continue

            masses.append(self.masses[i])
            x.append(self.x[i])
            y.append(self.y[i])
            r2d.append(self.r2d[i])
            r3d.append(self.r3d[i])
            mdefs.append(self.mdefs[i])

            massdefargs.append(self.mass_def_args[i])

        return np.array(masses), np.array(x), np.array(y), np.array(r2d), np.array(r3d), mdefs, np.array([z]*len(masses)), \
               massdefargs

    def convergence_at_z(self,z):

        m = self.mass_at_z(z)

        area = self.geometry._angle_to_arcsec_area(self.geometry._zlens, z)

        sigmacrit = self.geometry._lens_cosmo.get_sigmacrit(z)

        kappa = m / area / sigmacrit

        return kappa

    def mass_at_z(self,z):

        mass = 0

        for i, mi in enumerate(self.masses):
            if self.redshifts[i] == z:
                mass += mi

        return mass
