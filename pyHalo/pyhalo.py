from pyHalo.pyhalo_base import pyHaloBase
import numpy as np
from pyHalo.single_realization import Realization
from pyHalo.Rendering.Field.PowerLaw.powerlaw import LOSPowerLaw
from pyHalo.Rendering.Field.Delta.delta import LOSDelta
from pyHalo.Rendering.Main.mainlens import MainLensPowerLaw
from pyHalo.Rendering.render import render_los, render_main

class pyHalo(pyHaloBase):

    def __init__(self, zlens, zsource, cosmology_kwargs={},
                 kwargs_halo_mass_function={}):

        """
        This class manages the creation of dark matter substructure realizations, coordinating the
        rendering of line-of-sight and subhalos in the lensing volume. For usage examples see
        the example notebooks in pyhalo/example_notebooks

        :param zlens: lens redshift
        :param zsource: source redshift
        :param cosmology_kwargs:
        keyword arguments for 'Cosmology' class. See documentation in cosmology.py
        :param halo_mass_function_args:
        keyword arguments for 'LensingMassFunction' class. See documentation in lensing_mass_function.py
        :param kwargs_massfunc:
        keyword arguments
        """
        super(pyHalo, self).__init__(zlens, zsource, cosmology_kwargs, kwargs_halo_mass_function)

    def render(self, type, args, nrealizations=1, verbose=False):

        self.halo_mass_function = self.build_LOS_mass_function(args)
        self._geometry = self.halo_mass_function.geometry

        realizations = []

        for n in range(nrealizations):

            args = self._add_profile_params(args, False)

            realizations.append(self._render_single(type, args, verbose))

        return realizations

    def _render_single(self, type, args, verbose,
                       add_mass_sheet=True, x_center_lens=0., y_center_lens=0.):

        assert type in ['main_lens', 'composite_powerlaw', 'line_of_sight', 'dynamic_main', 'dynamic_LOS']

        flag = []
        init = True

        mass_sheet = add_mass_sheet

        lens_plane_redshifts, delta_zs = self.lens_plane_redshifts(args)

        rendering_classes = []

        if type == 'main_lens' or type == 'composite_powerlaw':

            rendering_class = MainLensPowerLaw(args, self._geometry, x_center_lens, y_center_lens)
            rendering_classes += [rendering_class]
            mdef = args['mdef_main']

            masses, x, y, r2d, r3d, redshifts = render_main(rendering_class)
            flag += [True] * len(masses)
            init = False
            mdefs = [mdef] * len(masses)

        if type == 'composite_powerlaw' or type == 'line_of_sight':

            if args['mass_func_type'] == 'DELTA':
                mass_sheet = False
                rendering_class = LOSDelta(args, self.halo_mass_function, self._geometry, args['log_mlow'],
                                           lens_plane_redshifts, delta_zs)
                rendering_classes += [rendering_class]

            elif args['mass_func_type'] == 'POWER_LAW':
                rendering_class = LOSPowerLaw(args, self.halo_mass_function, self._geometry, lens_plane_redshifts, delta_zs)
                rendering_classes += [rendering_class]

            else:
                raise Exception('Must specify mass_func_type.\nAllowed types: POWER_LAW, DELTA')

            mdef_los = args['mdef_los']

            if init:
                masses, x, y, r2d, r3d, redshifts \
                    = render_los(rendering_class, lens_plane_redshifts, delta_zs, args['zmin'], args['zmax'])
                flag += [False] * len(masses)
                mdefs = [mdef_los] * len(masses)

            else:

                field_halo_masses, field_xpos, field_ypos, field_r2d, field_r3d, field_z \
                    = render_los(rendering_class, lens_plane_redshifts, delta_zs, args['zmin'], args['zmax'])

                masses = np.append(masses, field_halo_masses)
                x = np.append(x, field_xpos)
                y = np.append(y, field_ypos)
                r2d = np.append(r2d, field_r2d)
                r3d = np.append(r3d, field_r3d)
                redshifts = np.append(redshifts, field_z)
                flag += [False] * len(field_halo_masses)
                mdefs += [mdef_los] * len(field_halo_masses)

        realization = Realization(masses, x, y, r2d, r3d, mdefs, redshifts, flag, self.halo_mass_function,
                                  other_params=args, mass_sheet_correction=mass_sheet,
                                  rendering_classes=rendering_classes)

        return realization







