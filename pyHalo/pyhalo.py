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
        :param kwargs_halo_mass_function:
        keyword arguments for 'LensingMassFunction' class. See documentation in lensing_mass_function.py
        :param cosmology_kwargs
        keyword arguments that specify cosmological parameters
        """
        self._cosmology_kwargs = cosmology_kwargs
        self._kwargs_mass_function = kwargs_halo_mass_function
        self._halo_mass_function_args = kwargs_halo_mass_function
        self.reset_redshifts(zlens, zsource)

    def lens_plane_redshifts(self, kwargs_render={}):

        """
        This routine sets up the redshift planes along the line of sight in the lens system
        :param kwargs_render: keyword arguments, if none are specified default values will be used (see defaults.py)
        :return: lens plane redshifts and the thickness of each slice
        """

        zmin = lenscone_default.default_zstart
        if 'zstep' not in kwargs_render.keys():
            zstep = lenscone_default.default_z_step
        else:
            zstep = kwargs_render['zstep']

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

    def render(self, model_list, mass_function_list, kwars_model_list, spatial_distribution_list, geometry, nrealizations=1):

        lens_cosmo = LensCosmo(self.zlens, self.zsource, self.cosmology)
        plane_redshifts, redshift_spacing = self.lens_plane_redshifts(keywords_master)
        realization_list = []

        for n in range(nrealizations):

            population_model = HaloPopulation(model_list, mass_function_list, spatial_distribution_list,
                 kwars_model_list, lens_cosmo, geometry, plane_redshifts, redshift_spacing)

            masses, x_arcsec, y_arcsec, r3d, redshifts, subhalo_flag = population_model.render()

            mdefs = []
            for i in range(0, len(masses)):
                if subhalo_flag[i]:
                    mdefs += [keywords_master['mdef_subs']]
                else:
                    mdefs += [keywords_master['mdef_los']]

            realization = Realization(masses, x_arcsec, y_arcsec, r3d, mdefs, redshifts, subhalo_flag, lens_cosmo,
                                      kwargs_realization=keywords_master, mass_sheet_correction=convergence_sheet_correction,
                                      rendering_classes=population_model.rendering_classes, geometry=geometry)
            realization_list.append(realization)

        return realization_list
