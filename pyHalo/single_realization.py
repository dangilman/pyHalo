import numpy as np
from pyHalo.Lensing.NFW import NFWLensing
from pyHalo.Lensing.TNFW import TNFWLensing
from pyHalo.Lensing.PTmass import PTmasslensing

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

    def lensing_quantities(self, mass_sheet_correction=True):

        kwargs_lens = []
        lens_model_names = []
        redshift_list = self.redshifts

        for i,mdef in enumerate(self.mdefs):

            args = {'x': self.x[i], 'y': self.y[i], 'mass': self.masses[i]}
            lens_model_names.append(mdef)

            if mdef == 'NFW':
                lensing = NFWLensing(self.lens_cosmo)
                args.update({'concentration':self.mass_def_args[i]['concentration'],'redshift':self.redshifts[i]})
            elif mdef == 'TNFW':
                lensing = TNFWLensing(self.lens_cosmo)
                args.update({'concentration': self.mass_def_args[i]['concentration'], 'redshift': self.redshifts[i]})
                args.update({'r_trunc': self.mass_def_args[i]['r_trunc']})
            elif mdef == 'PTmass':
                lensing = PTmasslensing(self.lens_cosmo)
                args.update({'redshift': self.redshifts[i]})
            else:
                raise ValueError('halo profile '+str(mdef)+' not recongnized.')

            kwargs_lens.append(lensing.params(**args))

        if mass_sheet_correction:

            kwargs_mass_sheets, z_sheets = self.mass_sheet_correction()

            kwargs_lens += kwargs_mass_sheets
            lens_model_names += ['CONVERGENCE'] * len(kwargs_mass_sheets)
            redshift_list += z_sheets

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

        inds = np.where(self.redshifts == z)
        masses = self.masses[inds]
        x = self.x[inds]
        y = self.y[inds]
        args = {}
        for key in self.mass_def_args.keys():
            args[key] = self.mass_def_args[key][inds]

        return masses, x, y, args

    def convergence_at_z(self,z):

        m = self.mass_at_z(z)

        area = self.geometry._angle_to_arcsec_area(z, self.geometry._zlens)

        sigmacrit = self.geometry._lens_cosmo.get_sigmacrit(z)

        return m / area / sigmacrit

    def mass_at_z(self,z):

        mass = 0

        for zi in self.redshifts:

            mass += np.sum(self.masses[np.where(z == zi)])

        return mass
