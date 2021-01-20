from pyHalo.Rendering.subhalos import Subhalos
from pyHalo.Rendering.line_of_sight import LineOfSight
from pyHalo.Rendering.two_halo import TwoHaloContribution
import numpy as np

class HaloPopulation(object):

    """
    This class handles combinations of different preset halo populations. Models currently implemented are LINE_OF_SIGHT,
    SUBHALOS, AND TWO_HALO. See the documentation of each of these classes for details about what kinds of halos it
    adds to the lens model.
    """

    def __init__(self, model_list, keywords_master, lens_cosmo, geometry, halo_mass_function=None,
                 lens_plane_redshift_list=None, redshift_spacings=None):

        """

        :param model_list: A list of population models (e.g. ['SUBHALOS', 'LINE_OF_SIGHT']
        :param keywords_master: a dictionary of keyword arguments to be passed to each model class
        :param lens_cosmo: an instance of LensCosmo (see Halos.lens_cosmo)
        :param geometry: an instance of Geometry (see Cosmology.geometry)
        :param halo_mass_function: an instance of LensingMassFunction (see Cosmology.lensing_mass_function)
        :param lens_plane_redshift_list: a list of redshifts at which to render halos
        :param redshift_spacings: a list of redshift increments between each lens plane (should be the same length as
        lens_plane_redshifts)
        """
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

        """
        Generates the masses, positions, 3D position inside host halo (if subhalo), and a list of bools that indicate
        whether each object is a subhalo

        :return: masses [M_sun], x coordinate [arcsec], y coordinate [arcsec], 3D position inside host halo [kpc],
        redshifts, list of bools indicating whether each object is a subhalo
        """
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

        """
        This routine combines the negative convergence sheet corrections corresponding to each halo population. This
        is necessary because adding dark matter profiles to a lens model and not subtracting the mean of what you've
        added effectively makes the Universe too dense in all of your simulations.

        :return: list of profiles, redshifts, and lenstronomy keyword arguments for the negative convergence sheet models
        """
        kwargs_list = []
        profile_list = []
        redshift_list = []

        for model in self.rendering_classes:

            kwargs, profile_names, redshifts = model.convergence_sheet_correction()
            kwargs_list += kwargs
            profile_list += profile_names
            redshift_list += redshifts

        return profile_list, redshift_list, kwargs_list




