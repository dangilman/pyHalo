from pyHalo.Rendering.subhalos import Subhalos
from pyHalo.Rendering.line_of_sight import LineOfSight
from pyHalo.Rendering.two_halo import TwoHaloContribution
import numpy as np

class HaloPopulation(object):

    halo_populations = ['LINE_OF_SIGHT', 'SUBHALOS', 'TWO_HALO']

    def __init__(self, model_list, keywords_master, lens_cosmo, geometry, halo_mass_function=None,
                 lens_plane_redshift_list=None, redshift_spacings=None):

        self.rendering_classes = []

        for population_model in model_list:
            if population_model == 'LINE_OF_SIGHT':
                model = LineOfSight(keywords_master, halo_mass_function, geometry, lens_cosmo,
                                    lens_plane_redshift_list, redshift_spacings)
            elif population_model == 'SUBHALOS':
                model = Subhalos(keywords_master, geometry, lens_cosmo)
            elif population_model == 'TWO_HALO':
                model = TwoHaloContribution(keywords_master, halo_mass_function, geometry, lens_cosmo,
                                            lens_plane_redshift_list, redshift_spacings)
            else:
                raise Exception('model '+str(population_model)+' not recognized. ')

            self.rendering_classes.append(model)

    def render(self):

        masses = np.array([])
        x = np.array([])
        y = np.array([])
        r3d = np.array([])
        redshifts = np.array([])
        is_subhalo_flag = []

        for model in self.rendering_classes:

            m, _x, _y, _r3d, _z, sub_flag = model.render()
            masses = np.append(masses, m)
            x = np.append(x, _x)
            y = np.append(y, _y)
            r3d = np.append(r3d, _r3d)
            redshifts = np.append(redshifts, _z)
            is_subhalo_flag += sub_flag

        return masses, x, y, r3d, redshifts, is_subhalo_flag

    def convergence_sheet_correction(self):

        kwargs_list = []
        profile_list = []
        redshift_list = []

        for model in self.rendering_classes:

            kwargs, profile_names, redshifts = model.convergence_sheet_correction()
            kwargs_list += kwargs
            profile_list += profile_names
            redshift_list += redshifts

        return profile_list, redshift_list, kwargs_list




