import numpy as np
from pyHalo.Halos.HaloModels.powerlaw import PowerLawSubhalo, PowerLawFieldHalo
from pyHalo.single_realization import Realization
from pyHalo.single_realization import realization_at_z
from pyHalo.Cosmology.lensing_mass_function import LensingMassFunction
from pyHalo.utilities import sample_circle
from pyHalo.utilities import sample_clustered


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

    def change_mass_definition(self, mdef, new_mdef, kwargs_new):

        kwargs_realization = self._realization._prof_params
        kwargs_realization.update(kwargs_new)
        halos = self._realization.halos
        new_halos = []

        if new_mdef == 'coreTNFW':
            from pyHalo.Halos.HaloModels.coreTNFW import coreTNFWSubhalo, coreTNFWFieldHalo
        else:
            raise Exception('changing to mass definition '+new_mdef + ' not implemented')

        for halo in halos:
            if halo.mdef == mdef:
                if halo.is_subhalo:
                    new_halo = coreTNFWSubhalo.fromTNFW(halo, kwargs_realization)
                else:
                    new_halo = coreTNFWFieldHalo.fromTNFW(halo, kwargs_realization)
                new_halos.append(new_halo)
            else:
                new_halos.append(halo)

        lens_cosmo = self._realization.lens_cosmo
        prof_params = self._realization._prof_params
        msheet_correction = self._realization._mass_sheet_correction
        rendering_classes = self._realization.rendering_classes
        rendering_center_x, rendering_center_y = self._realization.rendering_center

        return Realization.from_halos(new_halos, lens_cosmo, prof_params,
                                      msheet_correction, rendering_classes,
                                      rendering_center_x, rendering_center_y,
                                      self._realization.geometry)



    def find_core_collapsed_halos(self, time_scale_function, velocity_dispersion_function,
                                  cross_section, t_sub=10., t_field=100., t_sub_range=2, t_field_range=2.,
                                  model_type='TCHANNEL'):

        """
        :param time_scale_function: a function that computes the characteristic timescale for SIDM halos. This function
        must take as an input the NFW halo density normalization, velocity dispersion, and cross section class,
        and it must return a timescale (t_scale)
        :param velocity_dispersion_function: a function that computes the central velocity disperion of the halo

        It must be callable as:
        v = velocity_dispersion_function(halo_mass, redshift, delta_c_over_c, model_type, additional_keyword_arguments)

        where model_type is a string (see for example the function solve_sigmav_with_interpolation in sidmpy.py)

        :param velocity_averaged_cross_section_function: a function that returns the velocity averaged interaction
        cross section [cm^2 / gram * km/sec]
        :param t_sub: sets the timescale for subhalo core collapse; subhalos collapse at t_sub * t_scale
        :param t_field: sets the timescale for field halo core collapse; field halos collapse at t_field * t_scale
        :param t_sub_range: halos begin to core collapse (probability = 0) at t_sub - t_sub_range, and all are core
        collapsed by t = t_sub + t_sub_range (probability = 1)
        :param t_field_range: field halos begin to core collapse (probability = 0) at t_field - t_field_range, and all
        are core collapsed by t = t_field + t_field_range (probability = 1)
        :param model_type: specifies the cross section model to use when computing the solution to the velocity
        dispersion of the halo
        :return: indexes of halos that are core collapsed given
        """
        inds = []
        for i, halo in enumerate(self._realization.halos):

            if halo.mdef not in ['NFW', 'TNFW', 'coreTNFW']:
                continue

            # fit calibrated from the NFW velocity dispersion inside rs between 10^6 and 10^10
            # coeffs = [0.31575757, -1.74259129]
            # log_vrms = coeffs[0] * np.log10(halo.mass) + coeffs[1]
            # v_rms = 10 ** log_vrms

            concentration = halo.profile_args[0]
            rhos, rs = halo.params_physical['rhos'], halo.params_physical['rs']
            median_concentration = self._realization.lens_cosmo.NFW_concentration(halo.mass, halo.z, scatter=False)
            delta_c_over_c = 1 - concentration/median_concentration
            v_rms = velocity_dispersion_function(halo.mass, halo.z, delta_c_over_c, model_type, cross_section.kwargs)
            timescale = time_scale_function(rhos, v_rms, cross_section)

            if halo.is_subhalo:
                tcollapse_min = timescale * t_sub / t_sub_range
                tcollapse_max = timescale * t_sub * t_sub_range
            else:
                tcollapse_min = timescale * t_field / t_field_range
                tcollapse_max = timescale * t_field * t_field_range

            halo_age = self._realization.lens_cosmo.cosmo.halo_age(halo.z)

            if halo_age > tcollapse_max:
                p = 1.
            elif halo_age < tcollapse_min:
                p = 0.
            else:
                p = (halo_age - tcollapse_min) / (tcollapse_max - tcollapse_min)

            u = np.random.rand()
            if p >= u:
                inds.append(i)

        return inds

    def add_core_collapsed_halos(self, indexes, **kwargs_halo):

        """
        This function turns NFW halos in a realization into profiles modeled as PseudoJaffe profiles
        with 1/r^2 central density profiles with the same total mass as the original NFW
        profile.

        :param indexes: the indexes of halos in the realization to transform into PsuedoJaffe profiles
        :param log_slope_halo: the logarithmic slope of the collapsed halos
        :return: A new instance of Realization where the halos indexed by indexes
        in the original realization have their mass definitions changed to PsuedoJaffe
        """

        halos = self._realization.halos
        new_halos = []

        collapsed_subhalo_profile = PowerLawSubhalo
        collapsed_field_profile = PowerLawFieldHalo

        for i, halo in enumerate(halos):

            if i in indexes:

                halo._args.update(kwargs_halo)
                if halo.is_subhalo:
                    new_halo = collapsed_subhalo_profile(halo.mass, halo.x, halo.y, halo.r3d, collapsed_subhalo_profile,
                                                 halo.z, True, halo.lens_cosmo, halo._args, halo.unique_tag)
                else:
                    new_halo = collapsed_field_profile(halo.mass, halo.x, halo.y, halo.r3d, collapsed_field_profile,
                                                 halo.z, False, halo.lens_cosmo, halo._args, halo.unique_tag)
                new_halos.append(new_halo)

            else:
                new_halos.append(halo)

        lens_cosmo = self._realization.lens_cosmo
        prof_params = self._realization._prof_params
        msheet_correction = self._realization._mass_sheet_correction
        rendering_classes = self._realization.rendering_classes
        rendering_center_x, rendering_center_y = self._realization.rendering_center

        return Realization.from_halos(new_halos, lens_cosmo, prof_params,
                                      msheet_correction, rendering_classes,
                                      rendering_center_x, rendering_center_y,
                                      self._realization.geometry)
    
    def _pbh_in_lensplane (self, lens_plane_redshifts, cosmology, cosmo_geometry, max_rendering_range_base, 
                       x_interp_list, y_interp_list, image_index, LMF, mlow, mhigh, rho_DM, npix, M, realization):
        """
        This function generates primordial black hole parameters at each lensing plane.

        Parameters
        ----------
        lens_plane_redshifts : redshifts at which to render lensing          
        cosmology : pyhalo cosmology model    
        cosmo_geometry : pyhalo cosmo_geometry instance      
        max_rendering_range_base : radius of rendering area to be scaled by cosmology (arcsec)          
        x_interp_list : from pyhalo.utilities interpolate_ray_paths
        y_interp_list : from pyhalo.utilities interpolate_ray_paths
        image_index : which lens image is being modeled  
        LMF : pyhalo lensing mass function instance          
        mlow : lowest halo mass to render       
        mhigh : highest halo mass to render
        rho_DM : density of dark matter         
        npix : number of pixels on one axis   
        M : mass of PBH (solar masses)
        realization : the input pyhalo realization model
            

        Returns
        -------
        pbh_x_coordinates : primordial black hole x-coordinates (arcsec)
        pbh_y_coordinates : primordial black hole y-coordinates (arcsec)
        pbh_redshifts : redshift of each primordial black hole
        pbh_masses : mass of each primordial black hole
            

        """
        # intializing storage arrays
        pbh_masses = np.array([])
        pbh_x_coordinates = np.array([])
        pbh_y_coordinates = np.array([])
        pbh_redshifts = []
        
            
        for zi in lens_plane_redshifts:
            
            d_comoving = cosmology.D_C_z(zi)
            
            # this scales the rendering area with redshift and generates a grid
            max_rendering_range = max_rendering_range_base * cosmo_geometry.rendering_scale(zi) 
            
            
            # compute the angular coordinate around which to render halos
            center_x, center_y = x_interp_list[image_index](d_comoving), y_interp_list[image_index](d_comoving)
            
            # get the realization at redshift z
            realization_at_redshift, indexes = realization_at_z(realization, zi, center_x, center_y, max_rendering_range)
            
            # get an instance of LensModel at redshift z
            args = realization_at_redshift.lensing_quantities(add_mass_sheet_correction=False)
            lens_model_list_at_plane, kwargs_lens_at_plane = args[0], args[2]
            
            delta_zi = 0.02
            #delta_zi= lens_plane_redshifts[1]-lens_plane_redshifts[0]   
            V = cosmo_geometry.volume_element_comoving(zi, delta_z=delta_zi, radius=max_rendering_range)
          
            f_halos =  LMF.mass_fraction_in_halos(zi, mlow, mhigh) # this is where mlow and mhigh are necessary
            
            # M is black hole mass (solar masses)
            
            _nsmooth = f_halos*rho_DM*(V/M)
            _nclumpy = (1-f_halos)*rho_DM*(V/M)
            
            Nclumpy = int(np.random.poisson(_nclumpy))
            Nsmooth = int(np.random.poisson(_nsmooth))
            
    
            if Nsmooth > 0:
                coord_x_smooth, coord_y_smooth = sample_circle (max_rendering_range, Nsmooth, center_x, center_y)
                z_shifts_smooth = [zi] * len(coord_x_smooth)
                masses_smooth = np.array([M] * len(coord_x_smooth))
                pbh_x_coordinates = np.append(pbh_x_coordinates, coord_x_smooth)
                pbh_y_coordinates = np.append(pbh_y_coordinates, coord_y_smooth)
                pbh_redshifts += z_shifts_smooth
                pbh_masses = np.append(pbh_masses, masses_smooth)
            
            # we can only generate PBH around halos if there are halos, hence the second condition
            if Nclumpy > 0 and len(lens_model_list_at_plane): 
                coord_x_clumpy, coord_y_clumpy = sample_clustered (lens_model_list_at_plane, center_x, center_y, 
                     kwargs_lens_at_plane, Nclumpy, max_rendering_range, npix)
                masses_clumpy = np.array([M] * len(coord_x_clumpy))
                z_shifts_clumpy = [zi] * len(masses_clumpy)
                pbh_x_coordinates = np.append(pbh_x_coordinates, coord_x_clumpy)
                pbh_y_coordinates = np.append(pbh_y_coordinates, coord_y_clumpy)
                pbh_redshifts += z_shifts_clumpy
                pbh_masses = np.append(pbh_masses, masses_clumpy)
                
        return pbh_x_coordinates, pbh_y_coordinates, pbh_redshifts, pbh_masses
    
    def pbh_around_image(self, compfrac, M, image_index, realization, cosmology, lens_cosmo, cosmo_geometry, 
                     zlens, zsource, x_interp_list, y_interp_list, max_rendering_range_base, npix):
        """
        This function creates a realization with primordial black holes.

        Parameters
        ----------
        compfrac : component fraction (mass fraction) of pbh in dark matter
        M : mass of primordial black hole (solar masses)
        image_index : which lens image is being modeled       
        realization : the input pyhalo realization model
        cosmology : pyhalo cosmology model
        lens_cosmo : pyhalo lens_cosmo instance          
        cosmo_geometry : pyhalo cosmo_geometry instance
        zlens : redshift of lens
        zsource : redshift of source  
        x_interp_list : from pyhalo.utilities interpolate_ray_paths    
        y_interp_list : from pyhalo.utilities interpolate_ray_paths 
        max_rendering_range_base : radius of rendering area to be scaled by cosmology (arcsec)           
        npix : number of pixels on one axis
            

        Returns
        -------
        black_hole_realization : Instance of Realization with primordial black hole parameters included

        """
    
        lens_plane_redshifts = self._realization.unique_redshifts
        
        
        # defining things for later
        profile_args = self._realization._prof_params
        conangle = profile_args['cone_opening_angle'] #arcsec 
        LMF = LensingMassFunction(cosmology, zlens, zsource, cone_opening_angle=conangle)
        mlow = 10**profile_args['log_mlow']
        mhigh = 10**profile_args['log_mhigh']
        rho_DM = LMF.component_density(compfrac)
    
        pbh_x_coordinates, pbh_y_coordinates, pbh_redshifts, pbh_masses = self._pbh_in_lensplane (lens_plane_redshifts, 
                                                                cosmology, cosmo_geometry, max_rendering_range_base, 
                                                                x_interp_list, y_interp_list, image_index, LMF, mlow, mhigh, rho_DM, npix, M, self._realization)
                
        # r3d is the three dimensional position of a subhalo which is used to compute a truncation radius, 
        # this is irrelevant for PBH so we can just specify it with a dummy array of ones
        r3d = np.array([None] * len(pbh_masses))
        mdefs = ['PT_MASS'] * len(pbh_masses)
        sub_flags = [False]*len(pbh_masses)
    #    print('realization ' +str(image_index)+' contains '+str(len(pbh_masses))+ ' black holes')
        black_hole_realization = self._realization(pbh_masses, pbh_x_coordinates, pbh_y_coordinates, r3d,  mdefs, pbh_redshifts, sub_flags,
                                            lens_cosmo, mass_sheet_correction=False, kwargs_realization=profile_args)
        return black_hole_realization