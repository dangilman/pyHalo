import numpy as np
from pyHalo.Halos.HaloModels.powerlaw import PowerLawSubhalo, PowerLawFieldHalo
from pyHalo.Halos.HaloModels.generalized_nfw import GeneralNFWSubhalo, GeneralNFWFieldHalo
from pyHalo.single_realization import Realization
from pyHalo.Halos.HaloModels.gaussian import Gaussian
from pyHalo.Rendering.correlated_structure import CorrelatedStructure
from pyHalo.Rendering.MassFunctions.delta_function import DeltaFunction
from pyHalo.Cosmology.geometry import Geometry
from pyHalo.utilities import generate_lens_plane_redshifts
from pyHalo.Rendering.SpatialDistributions.uniform import Uniform
from copy import deepcopy


class RealizationExtensions(object):

    """
    This class supports operations that modify individual instances of the class Realization
    (see pyHalo.single_realization).
    """

    def __init__(self, realization):

        """

        :param realization: an instance of Realization
        """

        self._realization = realization

    def core_collapse_by_mass(self, mass_ranges_subhalos, mass_ranges_field_halos,
                              probabilities_subhalos, probabilities_field_halos,
                              kwargs_sub=None, kwargs_field=None):

        """
        This routine transforms some fraction of subhalos and field halos into core collapsed profiles
        in the specified mass ranges

        :param mass_ranges_subhalos: a list of lists specifying the log10 halo mass ranges for subhalos
        e.g. mass_ranges_subhalos = [[6, 8], [8, 10]]
        :param mass_ranges_field_halos: a list of lists specifying the halo mass ranges for field halos
        e.g. mass_ranges_subhalos = [[6, 8], [8, 10]]
        :param probabilities_subhalos: a list of lists specifying the fraction of subhalos in each mass
        range that core collapse
        e.g. probabilities_subhalos = [0.5, 1.] makes half of subhalos with mass 10^6 - 10^8 collapse, and
        100% of subhalos with mass between 10^8 and 10^10 collapse

        If the entries in probabilities subhalos is a function, i.e. you pass in [function_1, function_2]
        instead of [0.5, 1.0], they functions will be evaluated at z_eval, i.e. the fraction of collpsed
        halos becomes [function_1(z_eval), function_2(z_eval)]

        :param probabilities_field_halos: same as probabilities subhalos, but for the population of field halos
        :param kwargs_sub: list of keyword arguments for each function passed in mass_ranges_subhalos, if the entries in
        mass_ranges_subhalos are callable functions
        :param kwargs_field: list of keyword arguments for each function passed in mass_ranges_field_halos, if the entries in
        mass_ranges_field_halos are callable functions
        :return: indexes of core collapsed halos
        """

        assert len(mass_ranges_subhalos) == len(probabilities_subhalos)
        assert len(mass_ranges_field_halos) == len(probabilities_field_halos)

        indexes = []

        for i_halo, halo in enumerate(self._realization.halos):

            u = np.random.rand()

            if halo.is_subhalo:
                for i, mrange in enumerate(mass_ranges_subhalos):
                    if halo.mass >= 10**mrange[0] and halo.mass < 10**mrange[1]:
                        if callable(probabilities_subhalos[i]):
                            prob = probabilities_subhalos[i](halo.z, **kwargs_sub[i])
                        else:
                            prob = probabilities_subhalos[i]
                        if u <= prob:
                            indexes.append(i_halo)
                        break
            else:
                for i, mrange in enumerate(mass_ranges_field_halos):
                    if halo.mass >= 10**mrange[0] and halo.mass < 10**mrange[1]:
                        if callable(probabilities_field_halos[i]):
                            prob = probabilities_field_halos[i](halo.z, **kwargs_field[i])
                        else:
                            prob = probabilities_field_halos[i]
                        if u <= prob:
                            indexes.append(i_halo)
                        break
        return indexes

    def add_core_collapsed_halos(self, indexes, halo_profile='SPL_CORE',**kwargs_halo):

        """
        This function turns NFW halos in a realization into profiles modeled as PseudoJaffe profiles
        with 1/r^2 central density profiles with the same total mass as the original NFW
        profile.

        :param indexes: the indexes of halos in the realization to transform into powerlaw or generalized nfw profiles
        :param halo_profile: specifies whether to transform to powerlaw (SPL_CORE) or generalized_nfw profile (GNFW)
        :param kwargs_halo: the keyword arguments for the collapsed halo profile
        :return: A new instance of Realization where the halos indexed by indexes
        in the original realization have their mass definitions changed to PsuedoJaffe
        """

        halos = self._realization.halos
        new_halos = []

        if halo_profile == 'GNFW':
            subhalo_model = GeneralNFWSubhalo
            fieldhalo_model = GeneralNFWFieldHalo
        elif halo_profile == 'SPL_CORE':
            subhalo_model = PowerLawSubhalo
            fieldhalo_model = PowerLawFieldHalo
        else:
            raise Exception('only halo profile models GNFW and SPL_CORE '
                            'implemented for collapsed objects')

        for i, halo in enumerate(halos):
            if i in indexes:
                halo._args.update(kwargs_halo)
                if halo.is_subhalo:
                    new_halo = subhalo_model(halo.mass, halo.x, halo.y, halo.r3d,
                                                 halo.z, True, halo.lens_cosmo, halo._args,
                                             halo._truncation_class,
                                             halo._concentration_class,
                                             halo.unique_tag)
                else:
                    new_halo = fieldhalo_model(halo.mass, halo.x, halo.y, halo.r3d,
                                                 halo.z, False, halo.lens_cosmo, halo._args,
                                               halo._truncation_class,
                                               halo._concentration_class,
                                               halo.unique_tag)
                new_halos.append(new_halo)
            else:
                new_halos.append(halo)

        return Realization.from_halos(new_halos, self._realization.lens_cosmo, self._realization.kwargs_halo_model,
                                      self._realization.apply_mass_sheet_correction,
                                      self._realization.rendering_classes,
                                      self._realization._rendering_center_x, self._realization._rendering_center_y,
                                      self._realization.geometry)

    def add_ULDM_fluctuations(self, de_Broglie_wavelength, fluctuation_amplitude,
                              fluctuation_size, fluctuation_size_variance, n_cut, n_fluc_scale=1., shape='ring',
                              args={'rmin':0.9,'rmax':1.1}, rescale_fluc_amp=True):

        """
        This function adds gaussian fluctuations of the given de Broglie wavelength to a realization.

        :param de_Broglie_wavelength: de Broglie wavelength of the DM particle in kpc
        :param fluctuation_amplitude: Standard deviation of amplitude distribution in convergence units
        :param fluctuation_size: half the physical size of an indiviudal blob [kpc]
        (Blobs are modeled as Gaussians, so their diameter is ~2 * fluctuation_size)
        :param fluctuation_size_variance: scales the variance of the distribution of sizes of fluctuations,
        relative to lambda_dB
        :param shape: keyword argument for fluctuation geometry, can place fluctuations in a

            'ring', 'ellipse, or 'aperture' (centered about lensing images)

        :param args: properties of the given shape, must match

            'ring' : {'rmin': , 'rmax': } (radii within which to place fluctuations, rmin < rmax)
            'ellipse' : {'amin': , 'amax': , 'bmin': , 'bmax': , 'angle':} (major and minor axes within which to place fluctuations, amin < amax, bmin < bmax)
            'aperture' : {'x_images': , 'y_images':, 'aperture'} (list of x and y image coordinates and aperture radius)

            Note that for 'ellipse' the 'angle' parameter is the angle in radians at which to orient the ellipse relative to the positive x-axis.

        :param num_cut: integer number of fluctuations above which to start cancelling fluctuations
        :param rescale_fluc_amp: Boolean specifying whether re-scale fluctuation amplitudes by sqrt(n_cut / n), where n
        is the total number of fluctuations in the given area and n_cut (defined below) is the maximum number to generate
        """

        if (shape != 'ring') and (shape != 'ellipse') and (shape != 'aperture'): # check shape keyword
            raise Exception('shape must be ring or ellipse or aperture!')

        # get number of fluctuations
        n_flucs = _get_number_flucs(self._realization, de_Broglie_wavelength,
                                    fluctuation_size/de_Broglie_wavelength, n_fluc_scale, shape, args)

        # if zero fluctuations, return original realization

        if shape == 'aperture':
            for i in range(0, len(n_flucs)):
                if n_flucs[i] != 0:
                    break
            else:
                return self._realization

        else:
            if n_flucs == 0:
                return self._realization

        if rescale_fluc_amp:
            if shape == 'aperture' and np.mean(n_flucs) > n_cut:
                fluctuation_amplitude /= np.sqrt(np.mean(n_flucs) / n_cut)
                n_flucs = np.array([int(n_cut)] * 4)
            elif shape != 'aperture' and n_flucs > n_cut:
                fluctuation_amplitude /= np.sqrt(n_flucs / n_cut)
                n_flucs = int(n_cut)

        # create fluctuations
        fluctuations = _get_fluctuation_halos(self._realization,
                                              fluctuation_amplitude,
                                              fluctuation_size,
                                              fluctuation_size_variance,
                                              shape,
                                              n_flucs,
                                              args)

        # realization args
        lens_cosmo = self._realization.lens_cosmo
        msheet_correction = self._realization._mass_sheet_correction
        rendering_classes = self._realization.rendering_classes
        rendering_center_x, rendering_center_y = self._realization.rendering_center
        kwargs_halo_model = None
        # realization containing only fluctuations
        fluc_realization = Realization.from_halos(fluctuations, lens_cosmo, kwargs_halo_model,
                                      msheet_correction, rendering_classes,
                                      rendering_center_x, rendering_center_y,
                                      self._realization.geometry)

        # join realization to dark substructure realization
        return self._realization.join(fluc_realization)

    def add_correlated_structure(self, mass_function_model,
                                 kwargs_mass_function,
                                 mass_definition,
                                   x_image_interp_list,
                                   y_image_interp_list,
                                   arcsec_per_pixel, r_max,
                                 rescale_normalizations=True):

        """
        Adds structure along the line of sight with a spatial distribution that tracks the dark matter density at each
        lens plane

        """

        realization_copy = deepcopy(self._realization)
        lens_plane_redshifts, delta_z_list = generate_lens_plane_redshifts(self._realization.lens_cosmo.z_lens,
                                                                           self._realization.lens_cosmo.z_source)
        correlated_structure = CorrelatedStructure(mass_function_model, kwargs_mass_function,
                 self._realization.geometry, self._realization.lens_cosmo, lens_plane_redshifts,
                                                   delta_z_list, self._realization)

        for image_index, (x_image_interp, y_image_interp) in enumerate(zip(x_image_interp_list, y_image_interp_list)):
            masses, x, y, r3d, redshifts, subhalo_flag, rescale_indicies, rescale_factors = correlated_structure.render(
                r_max, x_image_interp, y_image_interp, arcsec_per_pixel)

            if rescale_normalizations:
                for i, index in enumerate(rescale_indicies):
                    realization_copy.halos[index].rescale_normalization(rescale_factors[i])

            mdefs = [mass_definition] * len(masses)
            kwargs_halo_model = {'truncation_model': None, 'concentration_model': None, 'kwargs_density_profile': {}}
            if image_index==0:
                realization_pbh = Realization(masses, x, y, r3d, mdefs, redshifts, subhalo_flag,
                                           self._realization.lens_cosmo, kwargs_halo_model=kwargs_halo_model)
            elif image_index>0:
                realization_pbh = realization_pbh.join(Realization(masses, x, y, r3d, mdefs, redshifts, subhalo_flag,
                                           self._realization.lens_cosmo, kwargs_halo_model=kwargs_halo_model))

        new_realization = realization_copy.join(realization_pbh)

        return new_realization

    def add_primordial_black_holes(self, pbh_mass_fraction, kwargs_pbh_mass_function, mass_fraction_in_halos,
                                   x_image_interp_list, y_image_interp_list, r_max_arcsec, arcsec_per_pixel=0.005,
                                   rescale_normalizations=True):

        """
        This routine renders populations of primordial black holes modeled as point masses along the line of sight.
        The population of objects includes a smoothly distributed component, and a component that is clustered according
        to the population of halos generated in the instance of Realization used to instantiate the class.

        :param pbh_mass_fraction: the mass fraction of dark matter contained in primordial black holes
        :param kwargs_pbh_mass_function: keyword arguments for the PBH mass function
        :param mass_fraction_in_halos: the fraction of dark matter mass contained in halos in the mass range
        used to generate the instance of realization used to instantiate the class
        :param x_image_interp_list: a list of interp1d functions that return the angular x coordinate of a light ray
        given a comoving distance
        :param y_image_interp_list: a list of interp1d functions that return the angular y coordinate of a light ray
        given a comoving distance
        :param r_max_arcsec: the radius of the rendering region in arcsec, here an array corresponding to the coordinates used for x_interp_list
        :param arcsec_per_pixel: the resolution of the grid used to compute the population of PBH whose spatial
        distribution tracks the dark matter density along the LOS specific by the instance of Realization used to
        instantiate the class
        :param rescale_normalizations: bool; whether or not to rescale the density profile of halos to account for the
        mass added in correlated structure
        :return: a new instance of Realization that contains primordial black holes modeled as point masses
        """
        mass_definition = 'PT_MASS'
        plane_redshifts = self._realization.unique_redshifts
        delta_z = []
        for i, zi in enumerate(plane_redshifts[0:-1]):
            delta_z.append(plane_redshifts[i + 1] - plane_redshifts[i])
        delta_z.append(self._realization.lens_cosmo.z_source - plane_redshifts[-1])

        mass_fraction_smooth = (1 - mass_fraction_in_halos) * pbh_mass_fraction
        mass_fraction_clumpy = pbh_mass_fraction * mass_fraction_in_halos

        masses = np.array([])
        xcoords = np.array([])
        ycoords = np.array([])
        redshifts = np.array([])
        for x_image_interp, y_image_interp, r_max in zip(x_image_interp_list, y_image_interp_list, r_max_arcsec):
            geometry = Geometry(self._realization.lens_cosmo.cosmo,
                                        self._realization.lens_cosmo.z_lens,
                                        self._realization.lens_cosmo.z_source,
                                        2 * r_max,
                                        'DOUBLE_CONE')
            for zi, delta_zi in zip(plane_redshifts, delta_z):

                d = self._realization.lens_cosmo.cosmo.D_C_transverse(zi)
                angle_x, angle_y = x_image_interp(d), y_image_interp(d)
                rendering_radius = r_max * geometry.rendering_scale(zi)
                spatial_distribution_model_smooth = Uniform(rendering_radius, geometry)

                if kwargs_pbh_mass_function['mass_function_type'] == 'DELTA':
                    rho_smooth = mass_fraction_smooth * self._realization.lens_cosmo.cosmo.rho_dark_matter_crit
                    volume = geometry.volume_element_comoving(zi, delta_zi)
                    mass_function_smooth = DeltaFunction(10 ** kwargs_pbh_mass_function['logM'],
                                                         volume, rho_smooth, draw_poisson=True)
                else:
                    raise Exception('no mass function type for PBH currently implemented besides DELTA')

                m_smooth = mass_function_smooth.draw()
                if len(m_smooth) > 0:
                    kpc_per_asec = geometry.kpc_per_arcsec(zi)
                    x_kpc, y_kpc = spatial_distribution_model_smooth.draw(len(m_smooth), zi,
                                                                          center_x=angle_x, center_y=angle_y)
                    x_arcsec, y_arcsec = x_kpc / kpc_per_asec, y_kpc / kpc_per_asec
                    masses = np.append(masses, m_smooth)
                    xcoords = np.append(xcoords, x_arcsec)
                    ycoords = np.append(ycoords, y_arcsec)
                    redshifts = np.append(redshifts, np.array([zi] * len(m_smooth)))

        mdefs = [mass_definition] * len(masses)
        r3d = np.array([None] * len(masses))
        subhalo_flag = [False] * len(masses)
        kwargs_halo_model = {'truncation_model': None, 'concentration_model': None, 'kwargs_density_profile': {}}
        realization_smooth = Realization(masses, xcoords, ycoords, r3d, mdefs, redshifts, subhalo_flag,
                                         self._realization.lens_cosmo, kwargs_halo_model=kwargs_halo_model,
                                         mass_sheet_correction=False)

        kwargs_pbh_mass_function['mass_fraction'] = mass_fraction_clumpy

        for ii, (x_image_interp, y_image_interp, r_max) in enumerate(zip(x_image_interp_list, y_image_interp_list, r_max_arcsec)):
            realization_with_clustering_temp = self.add_correlated_structure(DeltaFunction,
                                 kwargs_pbh_mass_function,
                                 mass_definition,
                                   [x_image_interp],
                                   [y_image_interp],
                                   arcsec_per_pixel, r_max,
                                 rescale_normalizations)
            if ii == 0:
                realization_with_clustering = realization_with_clustering_temp
                continue # first time through loop
            realization_with_clustering = realization_with_clustering.join(realization_with_clustering_temp)

        return realization_with_clustering.join(realization_smooth)

def _get_number_flucs(realization, de_Broglie_wavelength, fluctuation_size_scale, n_fluc_scale, shape, args):
    """
    This function returns the number of fluctuations to place in the realization.

    :param realization: the realization to which to add the fluctuations
    :param de_Broglie_wavelength: de Broglie wavelength of the DM particle in kpc
    :param fluctuation_size_scale: sets the size of the fluctuatiions relative to the de Broglie wavelength
    :param n_fluc_scale: rescales the total number of fluctuations
    :param shape: keyword argument for fluctuation geometry, see 'add_ULDM_fluctuations'
    :param args: properties of the given shape, see 'add_ULDM_fluctuations'
    """

    D_d = realization.lens_cosmo.cosmo.D_A_z(realization._zlens)
    arcsec = realization.lens_cosmo.cosmo.arcsec
    fluc_area=np.pi*(de_Broglie_wavelength * fluctuation_size_scale)**2 #de Broglie area
    to_kpc = D_d * arcsec * 1e3 # convert arcsec to kpc

    if shape=='ring': # fluctuations in a circular slice (for visualization purposes)

        rmin_kpc,rmax_kpc = args['rmin'] * to_kpc, args['rmax'] * to_kpc #args in kpc
        area_ring = np.pi*(rmax_kpc**2-rmin_kpc**2) # volume of ring
        n_flucs_expected=n_fluc_scale*area_ring/fluc_area # number of fluctuations in ring
        n_flucs = np.random.poisson(n_flucs_expected)

    if shape=='ellipse': # fluctuations in a elliptical slice (for visualization purposes)

        amin_kpc,bmin_kpc,amax_kpc,bmax_kpc=args['amin'] * to_kpc, args['bmin'] * to_kpc, args['amax'] * to_kpc, args['bmax'] * to_kpc #args in kpc
        area_ellipse=np.pi*(amax_kpc*bmax_kpc - amin_kpc*bmin_kpc) # volume of ellipse
        n_flucs_expected=n_fluc_scale*area_ellipse/fluc_area # number of fluctuations in ellipse
        n_flucs = np.random.poisson(n_flucs_expected)

    if shape=='aperture': # fluctuations around lensing images (for computation)

        n_images=len(args['x_images']) #number of lensed images
        r_kpc = args['aperture'] * to_kpc #aperture in kpc
        area_aperture = np.pi*r_kpc**2 # aperture area
        n_flucs_expected = n_fluc_scale*area_aperture/fluc_area #number of expected fluctuations per aperture
        n_flucs = np.random.poisson(n_flucs_expected,n_images) #draw number of fluctuations from poisson distribution for each image
        n_flucs = n_flucs[n_flucs!=0] #get rid of aperture if zero fluctuations within it

    return n_flucs

def _get_fluctuation_halos(realization, fluctuation_amplitude, fluctuation_size, fluctuation_size_variance, shape, n_flucs, args):
    """
    This function creates 'n_flucs' Gaussian fluctuations and places them according to 'shape'.

    :param realization: the realization to which to add the fluctuations
    :param fluctuation_amplitude: Standard deviation of amplitude distribution in convergence units
    :param fluctuation_size: half the physical size of a fluctuation (individual blobs are modeled as Gaussians, with
    a variance fluctuation_size)
    :param fluctuation_size_variance: scales the variance of the distribution of sizes of fluctuations, relative to lambda_dB
    :param rho_bar: typical convergence of a fluctuation at its peak
    :param shape: keyword argument for fluctuation geometry, see 'add_ULDM_fluctuations'
    :param n_flucs: Number of fluctuations to make
    :param args: properties of the given shape, see 'add_ULDM_fluctuations'
    """

    kpc_per_arcsec = realization.lens_cosmo.cosmo.kpc_proper_per_asec(realization._zlens)
    fluc_var_angle = fluctuation_size / kpc_per_arcsec # gaussian variance in arcsec
    fluctuation_size_variance_angle = fluctuation_size_variance / kpc_per_arcsec

    if shape != 'aperture':

        sigs = np.abs(np.random.normal(fluc_var_angle,fluctuation_size_variance_angle,n_flucs)) #random widths
        kappa0 = np.random.normal(0, fluctuation_amplitude, n_flucs)
        # kappa0 = amp / (2 * np.pi * sigma ** 2)
        amps = kappa0 * 2 * np.pi * sigs ** 2

    if shape=='ring':
        angles = np.random.uniform(0,2*np.pi,n_flucs)  # random angles
        radii = args['rmin'] + np.sqrt(np.random.uniform(0,1,n_flucs))*(args['rmax']-args['rmin']) #random radii
        xs = radii*np.cos(angles) #random x positions
        ys = radii*np.sin(angles) #random y positions

    if shape=='ellipse':
        angles = np.random.uniform(0,2*np.pi,n_flucs)  # random angles
        aa = np.sqrt(np.random.uniform(0,1,n_flucs))*(args['amax']-args['amin']) + args['amin'] #random axis 1
        bb = np.sqrt(np.random.uniform(0,1,n_flucs))*(args['bmax']-args['bmin']) + args['bmin'] #random axis 1
        xs = aa*np.cos(angles)*np.cos(args['angle'])-bb*np.sin(angles)*np.sin(args['angle']) #random x positions
        ys = aa*np.cos(angles)*np.sin(args['angle'])+bb*np.sin(angles)*np.cos(args['angle']) #random y positions

    if shape == 'aperture':

        amps, sigs, xs, ys = np.array([]), np.array([]), np.array([]), np.array([])

        for i in range(0, len(n_flucs)): #loop through each image

            sigs_i = np.random.normal(fluc_var_angle,fluctuation_size_variance_angle,n_flucs[i])
            sigs_i = np.absolute(sigs_i)

            kappa0 = np.random.normal(0, fluctuation_amplitude, n_flucs[i])
            amps_i = kappa0 * 2*np.pi*sigs_i**2

            angles_i = np.random.uniform(0, 2*np.pi, n_flucs[i])  # random angles
            r = np.random.uniform(0, args['aperture'] ** 2, int(n_flucs[i]))
            xs_i = r ** 0.5 * np.sin(angles_i) + args['x_images'][i]
            ys_i = r ** 0.5 * np.cos(angles_i) + args['y_images'][i]
            amps, sigs, xs, ys= np.append(amps, amps_i), np.append(sigs, sigs_i), np.append(xs, xs_i), np.append(ys, ys_i)

    args_fluc=[{'amp': amps[i], 'sigma': sigs[i], 'center_x': xs[i], 'center_y': ys[i]} for i in range(len(amps))]
    # kappa(r) = kappa * exp(-0.5 * r^2/sigma^2)
    #sigma_crit = realization.lens_cosmo.sigmacrit # in units M_sun / arcsec^2
    masses = np.absolute(amps)
    #masses = [10 ** 8.0] * len(xs)
    fluctuations = [Gaussian(masses[i], xs[i], ys[i], None, realization.lens_cosmo.z_lens,
                             True, realization.lens_cosmo, args_fluc[i],
                             truncation_class=None, concentration_class=None,
                             unique_tag=np.random.rand()) for i in range(len(amps))]

    return fluctuations
