from pyHalo.single_realization import Realization
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Rendering.halo_population import HaloPopulation
from pyHalo.utilities import generate_lens_plane_redshifts
from pyHalo.Halos.lens_cosmo import LensCosmo
from astropy.cosmology import FlatLambdaCDM
from pyHalo.defaults import cosmo_default


"""
This file contains top-level functions for generating halo realizations with pyHalo. Most users can avoid directly
working with these methods by loading pyHalo's "preset models" for halo substructure (see preset_models.py and the
classes in the PresetModels directory"
"""

_CODE_VERSION = '1.4.3'

def check_code_version(required):
    assert required == _CODE_VERSION

class pyHalo(object):
    CODE_VERSION = _CODE_VERSION
    """
    The main class used for generating realizations (see example notebook)
    """

    def __init__(self, zlens, zsource, kwargs_cosmo=None):

        """
        Initialize the class with the lens and source redshifts, and keyword arguments for the Cosmology class
        :param zlens: lens redshift (note: should be specified to 2 decimal places)
        :param zsource: source redshift
        :param kwargs_cosmo: keyword arguments for the Cosmology class
        """
        if kwargs_cosmo is None:
            astropy_instance = None
            kwargs_cosmo = cosmo_default.cosmo_param_dictionary
        else:
            # required keys for custom seteup:  'H0', 'Ob0', 'Om0', 'sigma8', 'flat', 'ns', 'power_law'
            astropy_instance = FlatLambdaCDM(H0=kwargs_cosmo['H0'],
                                             Ob0=kwargs_cosmo['Ob0'],
                                             Om0=kwargs_cosmo['Om0'])
        self.zlens = zlens
        self.zsource = zsource
        self.cosmology = Cosmology(astropy_instance, kwargs_cosmo)
        self.lens_cosmo = LensCosmo(self.zlens, self.zsource, self.cosmology)
        self.colossus_cosmo = self.cosmology.colossus

    @property
    def astropy_cosmo(self):
        """
        Returns the astropy cosmology instance for the pyHalo class instance
        """
        return self.cosmology.astropy

    def render(self, population_model_list, mass_function_class_list, kwargs_mass_function_list,
                   spatial_distribution_class_list, kwargs_spatial_distribution_list,
               geometry_class, mdef_subhalos, mdef_field_halos, kwargs_halo_model, two_halo_Lazar_correction=True,
               scale_2halo_boost_factor=1.0, nrealizations=1, lens_plane_spacing=None):

        """
        Return a list of instances of the SingleRealization class
        :param population_model_list: a list of pyHalo population models (see classes in Rendering)
        :param mass_function_class_list: a list of pyHalo mass functions (see classes in Rendering/MassFunctions)
        :param kwargs_mass_function_list: a list of keyword arguments for the mass functions
        :param spatial_distribution_class_list: a list of spatial distribution classes (see classes in
        Rendering/SpatialDistributions)
        :param kwargs_spatial_distribution_list: a list of keyword arguments for the spatial distributions
        :param geometry_class: an instance of the Geometry class (see Cosmology/geometry)
        :param mdef_subhalos: mass definition for subhalos
        :param mdef_field_halos: mass definition for field halos
        :param kwargs_halo_model: keyword arguments for the halo models
        :param two_halo_Lazar_correction: bool; include the two-halo term suggested by Lazar et al.
        :param scale_2halo_boost_factor: rescales the two-halo term by this factor
        :param nrealizations: number of realizations to generate
        :param lens_plane_spacing: spacing between line-of-sight lens planes, if None uses the default value LensConeDefaults
        :return: a list of instances of the SingleRealization class
        """
        realization_list = []
        for i in range(0, nrealizations):
            masses, x_arcsec, y_arcsec, r3d, redshifts, subhalo_flag, rendering_classes = self._render_masses_positions(population_model_list,
                                mass_function_class_list,
                                kwargs_mass_function_list,
                                spatial_distribution_class_list,
                                kwargs_spatial_distribution_list,
                                geometry_class, two_halo_Lazar_correction, scale_2halo_boost_factor, lens_plane_spacing)
            realization_list.append(self._create_realization(masses, x_arcsec, y_arcsec, r3d, redshifts, subhalo_flag, rendering_classes,
                                                            geometry_class, mdef_subhalos, mdef_field_halos, kwargs_halo_model))
        return realization_list

    def _render_masses_positions(self, population_model_list,
                                mass_function_class_list,
                                kwargs_mass_function,
                                spatial_distribution_class_list,
                                kwargs_spatial_distribution,
                                geometry_class,
                                two_halo_Lazar_correction=True,
                                scale_2halo_boost_factor=1.0,
                                lens_plane_spacing=None):

        """
        Generate halo masses, positions, redshifts, etc
        """

        plane_redshifts, redshift_spacing = generate_lens_plane_redshifts(self.zlens, self.zsource, lens_plane_spacing)
        population_model = HaloPopulation(population_model_list,
                                              mass_function_class_list,
                                              kwargs_mass_function,
                                              spatial_distribution_class_list,
                                              kwargs_spatial_distribution,
                                              self.lens_cosmo,
                                              geometry_class,
                                              plane_redshifts,
                                              redshift_spacing,
                                              two_halo_Lazar_correction,
                                              scale_2halo_boost_factor)
        masses, x_arcsec, y_arcsec, r3d, redshifts, subhalo_flag = population_model.render()
        return masses, x_arcsec, y_arcsec, r3d, redshifts, subhalo_flag, population_model.rendering_classes

    def _create_realization(self, masses, x_arcsec, y_arcsec, r3d, redshifts, subhalo_flag,
                           rendering_classes, geometry_class, mdef_subhalos, mdef_field_halos, kwargs_halo_model):
        """
        Generate a realization of halos
        """
        mdefs = []
        for i in range(0, len(masses)):
            if subhalo_flag[i]:
                mdefs += [mdef_field_halos]
            else:
                mdefs += [mdef_subhalos]
        realization = Realization(masses, x_arcsec, y_arcsec, r3d, mdefs, redshifts, subhalo_flag, self.lens_cosmo,
                                  kwargs_halo_model=kwargs_halo_model,
                                  mass_sheet_correction=True,
                                  rendering_classes=rendering_classes, geometry=geometry_class)
        return realization
