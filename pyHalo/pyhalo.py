from pyHalo.single_realization import Realization
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Rendering.halo_population import HaloPopulation
from pyHalo.utilities import generate_lens_plane_redshifts
from pyHalo.Halos.lens_cosmo import LensCosmo
from astropy.cosmology import FlatLambdaCDM
from pyHalo.defaults import cosmo_default

class pyHalo(object):

    """
    The main class used for generating realizations (see example notebook)
    """

    def __init__(self, zlens, zsource, kwargs_cosmo=None):

        """

        :param zlens:
        :param zsource:
        :param kwargs_cosmo:
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
        return self.cosmology.astropy

    def render(self, population_model_list, mass_function_class_list, kwargs_mass_function_list,
                   spatial_distribution_class_list, kwargs_spatial_distribution_list,
               geometry_class, mdef_subhalos, mdef_field_halos, kwargs_halo_model, nrealizations=1):

        """

        :param population_model_list:
        :param mass_function_class_list:
        :param kwargs_mass_function_list:
        :param spatial_distribution_class_list:
        :param kwargs_spatial_distribution_list:
        :param geometry_class:
        :param mdef_subhalos:
        :param mdef_field_halos:
        :param kwargs_halo_model:
        :param nrealizations:
        :return:
        """
        realization_list = []
        for i in range(0, nrealizations):
            masses, x_arcsec, y_arcsec, r3d, redshifts, subhalo_flag, rendering_classes = self.render_masses_positions(population_model_list,
                                mass_function_class_list,
                                kwargs_mass_function_list,
                                spatial_distribution_class_list,
                                kwargs_spatial_distribution_list,
                                geometry_class)
            realization_list.append(self.create_realization(masses, x_arcsec, y_arcsec, r3d, redshifts, subhalo_flag, rendering_classes,
                                                            geometry_class, mdef_subhalos, mdef_field_halos, kwargs_halo_model))
        return realization_list

    def render_masses_positions(self, population_model_list,
                                mass_function_class_list,
                                kwargs_mass_function,
                                spatial_distribution_class_list,
                                kwargs_spatial_distribution,
                                geometry_class):

        """

        :param population_model_list:
        :param mass_function_class_list:
        :param kwargs_mass_function:
        :param spatial_distribution_class_list:
        :param kwargs_spatial_distribution:
        :param geometry_class:
        :return:
        """

        plane_redshifts, redshift_spacing = generate_lens_plane_redshifts(self.zlens, self.zsource)
        population_model = HaloPopulation(population_model_list,
                                              mass_function_class_list,
                                              kwargs_mass_function,
                                              spatial_distribution_class_list,
                                              kwargs_spatial_distribution,
                                              self.lens_cosmo,
                                              geometry_class,
                                              plane_redshifts,
                                              redshift_spacing)
        masses, x_arcsec, y_arcsec, r3d, redshifts, subhalo_flag = population_model.render()
        return masses, x_arcsec, y_arcsec, r3d, redshifts, subhalo_flag, population_model.rendering_classes

    def create_realization(self, masses, x_arcsec, y_arcsec, r3d, redshifts, subhalo_flag,
                           rendering_classes, geometry_class, mdef_subhalos, mdef_field_halos, kwargs_halo_model):
        """

        :param masses:
        :param x_arcsec:
        :param y_arcsec:
        :param r3d:
        :param redshifts:
        :param subhalo_flag:
        :param rendering_classes:
        :param geometry_class:
        :param mdef_subhalos:
        :param mdef_field_halos:
        :param convergence_sheet_correction:
        :return:
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
