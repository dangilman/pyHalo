import numpy as np
from pyHalo.single_realization import Realization
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Rendering.halo_population import HaloPopulation
from pyHalo.defaults import lenscone_default
from pyHalo.Halos.lens_cosmo import LensCosmo


class pyHalo(object):

    """
    The main class used for generating realizations (see example notebook)
    """

    def __init__(self, zlens, zsource, cosmology_kwargs={}):

        """
        This class manages the creation of dark matter substructure realizations, coordinating the
        rendering of line-of-sight and subhalos in the lensing volume. For usage examples see
        the example notebooks in pyhalo/example_notebooks

        :param zlens: lens redshift
        :param zsource: source redshift
        :param cosmology_kwargs:
        keyword arguments for 'Cosmology' class. See documentation in cosmology.py
        :param kwargs_halo_mass_function:
        keyword arguments for 'LensingMassFunction' class. See documentation in lensing_mass_function.py
        :param cosmology_kwargs
        keyword arguments that specify cosmological parameters
        """
        self._cosmology_kwargs = cosmology_kwargs
        self.reset_redshifts(zlens, zsource)
        self._lens_cosmo = LensCosmo(self.zlens, self.zsource, self.cosmology)

    @property
    def lens_plane_redshifts(self):

        """
        This routine sets up the redshift planes along the line of sight in the lens system
        :param kwargs_render: keyword arguments, if none are specified default values will be used (see defaults.py)
        :return: lens plane redshifts and the thickness of each slice
        """

        zmin = lenscone_default.default_zstart
        zstep = lenscone_default.default_z_step

        front_z = np.arange(zmin, self.zlens, zstep)
        back_z = np.arange(self.zlens, self.zsource, zstep)
        redshifts = np.append(front_z, back_z)

        delta_zs = []
        for i in range(0, len(redshifts) - 1):
            delta_zs.append(redshifts[i + 1] - redshifts[i])
        delta_zs.append(self.zsource - redshifts[-1])

        return list(np.round(redshifts, 2)), np.round(delta_zs, 2)

    def reset_redshifts(self, zlens, zsource):

        self.zlens = zlens
        self.zsource = zsource
        self.cosmology = Cosmology(**self._cosmology_kwargs)
        self.halo_mass_function = None
        self.geometry = None

    @property
    def astropy_cosmo(self):
        return self.cosmology.astropy

    def render(self, population_model_list,
                                mass_function_class_list,
                                kwargs_mass_function,
                                spatial_distribution_class_list,
                                kwargs_spatial_distribution,
                                geometry_class, mdef_subhalos, mdef_field_halos, kwargs_halo_model, nrealizations=1):

        """

        :param population_model_list:
        :param mass_function_class_list:
        :param kwargs_mass_function:
        :param spatial_distribution_class_list:
        :param kwargs_spatial_distribution:
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
                                kwargs_mass_function,
                                spatial_distribution_class_list,
                                kwargs_spatial_distribution,
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

        plane_redshifts, redshift_spacing = self.lens_plane_redshifts
        population_model = HaloPopulation(population_model_list,
                                              mass_function_class_list,
                                              kwargs_mass_function,
                                              spatial_distribution_class_list,
                                              kwargs_spatial_distribution,
                                              self._lens_cosmo,
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
        realization = Realization(masses, x_arcsec, y_arcsec, r3d, mdefs, redshifts, subhalo_flag, self._lens_cosmo,
                                  kwargs_halo_model=kwargs_halo_model,
                                  mass_sheet_correction=True,
                                  rendering_classes=rendering_classes, geometry=geometry_class)
        return realization
