from pyHalo.Rendering.subhalos import Subhalos
from pyHalo.Rendering.line_of_sight import LineOfSight, LineOfSightNoSheet
from pyHalo.Rendering.two_halo import TwoHaloContribution
import numpy as np

class HaloPopulation(object):

    """
    This class handles combinations of different preset halo populations. Models currently implemented are LINE_OF_SIGHT,
    SUBHALOS, AND TWO_HALO. See the documentation of each of these classes for details about what kinds of halos it
    adds to the lens model.
    """

    def __init__(self, model_list, mass_function_class_list, kwargs_mass_function_list,
                 spatial_distribution_class_list, kwargs_spatial_distribution,
                 lens_cosmo, geometry,
                 lens_plane_redshift_list=None, redshift_spacings=None):

        """

        :param model_list: a list of names for different halo populations (e.g. ['LINE_OF_SIGHT', 'SUBHALOS', ...])
        :param mass_function_class_list: a list of non-instatiated mass function classes (see Rendering/MassFunctions)
        :param kwargs_mass_function_list: keyword arguments for the mass function classes
        :param spatial_distribution_class_list: a list of non-instantiated spatial distribution classes
        (see Rendering/SpatialDistributions)
        :param kwargs_spatial_distribution: keyword arguments for the spatial distribution classes
        :param lens_cosmo: an instance of LensCosmo (see Halos.lens_cosmo)
        :param geometry: an instance of Geometry class (see Cosmology.geometry)
        :param lens_plane_redshift_list: redshifts at which to render line-of-sight halos
        :param redshift_spacings: spacing between redshift planes
        """

        self.rendering_classes = []
        for i, model_name in enumerate(model_list):

            mass_function_model_class = mass_function_class_list[i]
            kwargs_model = kwargs_mass_function_list[i]

            if model_name == 'SUBHALOS':
                spatial_distribution_model = spatial_distribution_class_list[i].from_Mhost(**kwargs_spatial_distribution[i])
            else:
                spatial_distribution_model = spatial_distribution_class_list[i](**kwargs_spatial_distribution[i])

            if model_name == 'LINE_OF_SIGHT':
                model = LineOfSight(mass_function_model_class, kwargs_model, spatial_distribution_model,
                 geometry, lens_cosmo, lens_plane_redshift_list, redshift_spacings)
            elif model_name == 'LINE_OF_SIGHT_NOSHEET':
                model = LineOfSightNoSheet(mass_function_model_class, kwargs_model, spatial_distribution_model,
                 geometry, lens_cosmo, lens_plane_redshift_list, redshift_spacings)
            elif model_name == 'SUBHALOS':
                model = Subhalos(mass_function_model_class, kwargs_model, spatial_distribution_model,
                 geometry, lens_cosmo)
            elif model_name == 'TWO_HALO':
                model = TwoHaloContribution(mass_function_model_class, kwargs_model, spatial_distribution_model,
                 geometry, lens_cosmo, lens_plane_redshift_list, redshift_spacings)
            else:
                raise Exception('model '+str(model_name)+' not recognized. ')

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

    def convergence_sheet_correction(self, kwargs_mass_sheets=None):

        """
        This routine combines the negative convergence sheet corrections corresponding to each halo population. This
        is necessary because adding dark matter profiles to a lens model and not subtracting the mean of what you've
        added effectively makes the Universe too dense in all of your simulations.

        :param kwargs_mass_sheets: keyword arguments for the mass sheet correction
        :return: list of profiles, redshifts, and lenstronomy keyword arguments for the negative convergence sheet models
        """
        kwargs_list = []
        profile_list = []
        redshift_list = []

        for model in self.rendering_classes:

            kwargs, profile_names, redshifts = model.convergence_sheet_correction(kwargs_mass_sheets)
            kwargs_list += kwargs
            profile_list += profile_names
            redshift_list += redshifts

        return profile_list, redshift_list, kwargs_list




