import numpy as np
from copy import deepcopy
from pyHalo.Halos.HaloModels.powerlaw import PowerLawSubhalo, PowerLawFieldHalo
from pyHalo.single_realization import Realization
from pyHalo.Halos.HaloModels.gaussian import Gaussian
from pyHalo.defaults import RealizationDefaults
from lenstronomy.LensModel.Profiles.gaussian_kappa import GaussianKappa
from lenstronomy.LensModel.Profiles.nfw import NFW
import random
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree

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

    def core_collapse_by_mass(self, mass_ranges_subhalos, mass_ranges_field_halos,
                              probabilities_subhalos, probabilities_field_halos):

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
        :param probabilities_field_halos: same as probabilities subhalos, but for the population of field halos
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
                        if u <= probabilities_subhalos[i]:
                            indexes.append(i_halo)
                        break

            else:
                for i, mrange in enumerate(mass_ranges_field_halos):
                    if halo.mass >= 10**mrange[0] and halo.mass < 10**mrange[1]:
                        if u <= probabilities_field_halos[i]:
                            indexes.append(i_halo)
                        break

        return indexes

    def find_core_collapsed_halos(self, time_scale_function, velocity_dispersion_function,
                                  cross_section, t_sub=10., t_field=100., t_sub_range=2, t_field_range=2.,
                                  model_type='TCHANNEL'):

        """
        :param time_scale_function: a function that computes the characteristic timescale for SIDM halos. This function
        must take as an input the NFW halo density normalization, velocity dispersion, and cross section class,

        t_scale = time_scale_function(rhos, v_rms, cross_section_class)

        :param velocity_dispersion_function: a function that computes the central velocity disperion of the halo

        It must be callable as:
        v = velocity_dispersion_function(halo_mass, redshift, delta_c_over_c, model_type, additional_keyword_arguments)

        where model_type is a string (see for example the function solve_sigmav_with_interpolation in sidmpy.py)
        :param cross_section: the cross section class (see SIDMpy for examples)
        :param t_sub: sets the timescale for subhalo core collapse; subhalos collapse at t_sub * t_scale
        :param t_field: sets the timescale for field halo core collapse; field halos collapse at t_field * t_scale
        :param t_sub_range: halos begin to core collapse (probability = 0) at t_sub - t_sub_range, and all are core
        collapsed by t = t_sub + t_sub_range (probability = 1)
        :param t_field_range: field halos begin to core collapse (probability = 0) at t_field - t_field_range, and all
        are core collapsed by t = t_field + t_field_range (probability = 1)
        :param model_type: specifies the cross section model to use when computing the solution to the velocity
        dispersion of the halo
        :return: indexes of halos that are core collapsed
        """
        inds = []
        for i, halo in enumerate(self._realization.halos):

            if halo.mdef not in ['NFW', 'TNFW', 'coreTNFW']:
                continue

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
        :param kwargs_halo: the keyword arguments for the collapsed halo profile
        :return: A new instance of Realization where the halos indexed by indexes
        in the original realization have their mass definitions changed to PsuedoJaffe
        """

        halos = self._realization.halos
        new_halos = []

        for i, halo in enumerate(halos):

            if i in indexes:

                halo._args.update(kwargs_halo)
                if halo.is_subhalo:
                    new_halo = PowerLawSubhalo(halo.mass, halo.x, halo.y, halo.r3d, 'SPL_CORE',
                                                 halo.z, True, halo.lens_cosmo, halo._args, halo.unique_tag)
                else:
                    new_halo = PowerLawFieldHalo(halo.mass, halo.x, halo.y, halo.r3d, 'SPL_CORE',
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

    def add_ULDM_fluctuations(self,de_Broglie_wavelength,rho_mean=0.25,shape='ring',args={'rmin':0.9,'rmax':1.1}):

        """
        This function adds gaussian fluctuations of the given de Broglie wavelength to a realization.
        
        :param de_Broglie_wavelength: de Broglie wavelength of the DM particle in kpc
        :param rho_mean: typical convergence of a fluctuation at its peak
        :param shape: keyword argument for fluctuation geometry, can place fluctuations in a 
        
            'ring', 'ellipse, or 'aperture' (centered about lensing images)

        :param args: properties of the given shape, must match

            'ring' : {'rmin': , 'rmax': } (radii within which to place fluctuations, rmin < rmax)
            'ellipse; : {'amin': , 'amax': , 'bmin': , 'bmax': } (major and minor axes within which to place fluctuations, amin < amax, bmin < bmax)
            'aperture' : {'x_images': , 'y_images':, 'aperture'} (list of x and y image coordinates and aperture radius)

        :param num_cut: integer number of fluctuations above which to start Central Limit Theorem averaging approximation, if None no approximation
                        Warning: setting num_cut=None for a large number of fluctuations will take a while
        """

        if (shape != 'ring') and (shape != 'ellipse') and (shape != 'aperture'): # check shape keyword 

            raise Exception('shape must be ring or ellipse or aperture!')
        
        # create n_flucs fluctuations
        n_flucs = _get_number_flucs(self._realization,de_Broglie_wavelength,shape,args)
        fluctuations = _get_fluctuation_halos(self._realization,de_Broglie_wavelength,rho_mean,shape,n_flucs,args)

        # realization args
        lens_cosmo = self._realization.lens_cosmo
        prof_params = self._realization._prof_params
        msheet_correction = self._realization._mass_sheet_correction
        rendering_classes = self._realization.rendering_classes
        rendering_center_x, rendering_center_y = self._realization.rendering_center

        # realization containing only fluctuations
        fluc_realization = Realization.from_halos(fluctuations, lens_cosmo, prof_params,
                                      msheet_correction, rendering_classes,
                                      rendering_center_x, rendering_center_y,
                                      self._realization.geometry)

        # join realization to dark substructure realization
        return self._realization.join(fluc_realization)

def _get_number_flucs(realization,de_Broglie_wavelength,shape,args):
    """
    This function returns the number of fluctuations to place in the realization.

    :param realization: the realization to which to add the fluctuations
    :param de_Broglie_wavelength: de Broglie wavelength of the DM particle in kpc
    :param shape: keyword argument for fluctuation geometry, see 'add_ULDM_fluctuations'
    :param args: properties of the given shape, see 'add_ULDM_fluctuations'
    """

    D_d = realization.lens_cosmo.cosmo.D_A_z(realization._zlens)
    arcsec = realization.lens_cosmo.cosmo.arcsec
    fluc_area=np.pi*de_Broglie_wavelength**2 #de Broglie area
    to_kpc = D_d * arcsec * 1e3 # convert arcsec to kpc

    if shape=='ring': # fluctuations in a circular slice (for visualization purposes)

        rmin_kpc,rmax_kpc = args['rmin'] * to_kpc, args['rmax'] * to_kpc #args in kpc
        area_ring = np.pi*(rmax_kpc**2-rmin_kpc**2) # volume of ring
        n_flucs=int(area_ring/fluc_area) # number of fluctuations in ring
    
    if shape=='ellipse': # fluctuations in a elliptical slice (for visualization purposes)
        
        amin_kpc,bmin_kpc,amax_kpc,bmax_kpc=args['amin'] * to_kpc, args['bmin'] * to_kpc, args['amax'] * to_kpc, args['bmax'] * to_kpc #args in kpc
        area_ellipse=np.pi*(amax_kpc*bmax_kpc - amin_kpc*bmin_kpc) # volume of ellipse
        n_flucs=int(area_ellipse/fluc_area) # number of fluctuations in ellipse

    if shape=='aperture': # fluctuations around lensing images (for computation)

        n_images=len(args['x_images']) #number of lensed images
        r_kpc = args['aperture'] * D_d * arcsec * 1e3 #aperture in kpc
        area_aperture = np.pi*r_kpc**2 # aperture area
        n_flucs_expected = area_aperture/fluc_area #number of expected fluctuations per aperture
        n_flucs = np.random.poisson(n_flucs_expected,n_images) #draw number of fluctuations from poisson distribution for each image
        n_flucs[np.where(n_flucs == 0)] = 1 #avoid zero fluctuations

    return n_flucs

def _get_fluctuation_halos(realization,de_Broglie_wavelength,rho_bar,shape,n_flucs,args):
    """
    This function creates 'n_flucs' Gaussian fluctuations and places them according to 'shape'.

    :param realization: the realization to which to add the fluctuations
    :param de_Broglie_wavelength: de Broglie wavelength of the DM particle in kpc
    :param rho_bar: typical convergence of a fluctuation at its peak
    :param shape: keyword argument for fluctuation geometry, see 'add_ULDM_fluctuations'
    :param n_flucs: Number of fluctuations to make
    :param args: properties of the given shape, see 'add_ULDM_fluctuations'
    """

    D_d = realization.lens_cosmo.cosmo.D_A_z(realization._zlens)
    arcsec = realization.lens_cosmo.cosmo.arcsec
    r_dB_angle = de_Broglie_wavelength / D_d / arcsec / 1e3 #de Broglie wavelength in arcsec

    mass = RealizationDefaults().host_m200
    c = realization.lens_cosmo.NFW_concentration(mass, realization._zlens)
    _,_,r200 = realization.lens_cosmo.NFW_params_physical(mass,c,realization._zlens)
    CLT_factor = np.sqrt(2*de_Broglie_wavelength/3/r200)

    if shape!='aperture':

        sigs = np.abs(np.random.normal(r_dB_angle,r_dB_angle/2,n_flucs)) #random widths
        amps = CLT_factor*np.random.normal(0,np.sqrt(2)*rho_bar,n_flucs)*(2*np.pi*sigs**2) #random amplitudes

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

    if shape=='aperture':
        
        amps,sigs,xs,ys=np.array([]),np.array([]),np.array([]),np.array([])

        for i in range(len(n_flucs)): #loop through each image

            sigs_i = np.abs(np.random.normal(r_dB_angle,r_dB_angle/2,n_flucs[i])) #random widths
            amps_i = CLT_factor*np.random.normal(0,np.sqrt(2)*rho_bar,n_flucs[i])*(2*np.pi*sigs_i**2) #random amplitudes
            angles_i = np.random.uniform(0,2*np.pi,n_flucs[i])  # random angles
            radii_i = np.sqrt(np.random.uniform(0,1,n_flucs[i]))*args['aperture'] #random radii
            xs_i = radii_i*np.cos(angles_i)+args['x_images'][i] #random x positions
            ys_i = radii_i*np.sin(angles_i)+args['y_images'][i] #random y positions

            amps,sigs,xs,ys=np.append(amps,amps_i),np.append(sigs,sigs_i),np.append(xs,xs_i),np.append(ys,ys_i)

    args_fluc=[{'amp': amps[i], 'sigma': sigs[i], 'center_x': xs[i], 'center_y': ys[i]} for i in range(len(amps))]
    fluctuations = [Gaussian(5*sigs[i], xs[i], ys[i], None, None, realization._zlens, None, realization.lens_cosmo,args_fluc[i],np.random.rand()) for i in range(len(amps))]
    
    return fluctuations