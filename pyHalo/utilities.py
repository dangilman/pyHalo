import numpy as np
from pyHalo.defaults import lenscone_default
from scipy.interpolate import interp1d
from pyHalo.Cosmology.cosmology import Cosmology
from lenstronomy.LensModel.lens_model import LensModel

def interpolate_ray_paths(x_coordinates, y_coordinates, lens_model, kwargs_lens, zsource,
                          terminate_at_source=False, source_x=None, source_y=None, evaluate_at_mean=False,
                          cosmo=None):

    """
    :param x_coordinates: x coordinates to interpolate (arcsec) (list)
    :param y_coordinates: y coordinates to interpolate (arcsec) (list)
    Typically x_coordinates/y_coordinates would be four image positions, or the coordinate of the lens centroid
    :param lens_model: instance of LensModel (lenstronomy)
    :param kwargs_lens: keyword arguments for lens model
    :param zsource: source redshift
    :param terminate_at_source: fix the final angular coordinate to the source coordinate
    :param source_x: source x coordinate (arcsec)
    :param source_y: source y coordinate (arcsec)
    :param evaluate_at_mean: if True, returns two single interp1d instances (one for each of x/y) that return the
    average of each individual x/y coordinate evaluated at each lens plane. For example, if you pass in four images positions
    the output would be an interpolation of the average x/y coordinate along the path traversed by the light
    (This is useful for aligning realizations with a background source significantly offset from the lens centroid)

    :return: Instances of interp1d (scipy) that return the angular coordinate of a ray given a
    comoving distance
    """

    angle_x = []
    angle_y = []

    if cosmo is None:
        cosmo = Cosmology()

    for i, (xpos, ypos) in enumerate(zip(x_coordinates, y_coordinates)):

        theta_x = [xpos]
        theta_y = [ypos]

        ray_x, ray_y, d = compute_comoving_ray_path(xpos, ypos, lens_model, kwargs_lens, zsource,
                                                          terminate_at_source, source_x, source_y, cosmo=cosmo)

        for rx, ry, di in zip(ray_x[1:], ray_y[1:], d[1:]):
            theta_x.append(rx / di)
            theta_y.append(ry / di)

        distances = [0.] + list(d[1:])
        distances = np.array(distances)
        theta_x = np.array(theta_x)
        theta_y = np.array(theta_y)

        angle_x.append(interp1d(distances, theta_x))
        angle_y.append(interp1d(distances, theta_y))

    if evaluate_at_mean:

        zrange = np.linspace(0., zsource, 100)
        comoving_distance_calc = cosmo.D_C_transverse
        distances = [comoving_distance_calc(zi) for zi in zrange]

        angular_coordinates_x = []
        angular_coordinates_y = []
        for di in distances:
            x_coords = [ray_x(di) for ray_x in angle_x]
            y_coords = [ray_y(di) for ray_y in angle_y]
            x_center = np.mean(x_coords)
            y_center = np.mean(y_coords)
            angular_coordinates_x.append(x_center)
            angular_coordinates_y.append(y_center)

        angle_x = [interp1d(distances, angular_coordinates_x)]
        angle_y = [interp1d(distances, angular_coordinates_y)]

    return angle_x, angle_y

def compute_comoving_ray_path(x_coordinate, y_coordinate, lens_model, kwargs_lens, zsource,
                              terminate_at_source=False, source_x=None, source_y=None, cosmo=None):

        """
        :param x_coordinate: x coordinates to interpolate (arcsec) (float)
        :param y_coordinate: y coordinates to interpolate (arcsec) (float)
        Typically x_coordinates/y_coordinates would be four image positions, or the coordinate of the lens centroid
        :param lens_model: instance of LensModel (lenstronomy)
        :param kwargs_lens: keyword arguments for lens model
        :param zsource: source redshift
        :param terminate_at_source: fix the final angular coordinate to the source coordinate
        :param source_x: source x coordinate (arcsec)
        :param source_y: source y coordinate (arcsec)
        :return: Instance of interp1d (scipy) that returns the angular coordinate of a ray given a
        comoving distance
        """

        if cosmo is None:
            cosmo = Cosmology()

        redshift_list = lens_model.redshift_list + [zsource]
        zstep = lenscone_default.default_z_step
        finely_sampled_redshifts = np.linspace(zstep, zsource - zstep, 50)
        all_redshifts = np.unique(np.append(redshift_list, finely_sampled_redshifts))

        all_redshifts_sorted = all_redshifts[np.argsort(all_redshifts)]

        comoving_distance_calc = cosmo.D_C_transverse

        x_start, y_start = 0., 0.
        z_start = 0.

        x_list = [0.]
        y_list = [0.]
        distances = [0.]

        alpha_x_start, alpha_y_start = x_coordinate, y_coordinate
        for zi in all_redshifts_sorted:

            x_start, y_start, alpha_x_start, alpha_y_start = lens_model.lens_model.ray_shooting_partial(x_start, y_start,
                                                                                alpha_x_start, alpha_y_start,
                                                                                z_start, zi, kwargs_lens)
            d = float(comoving_distance_calc(zi))
            x_list.append(x_start)
            y_list.append(y_start)
            distances.append(d)
            z_start = zi

        if terminate_at_source:
            d_src = comoving_distance_calc(zsource)
            x_list[-1] = source_x * d_src
            y_list[-1] = source_y * d_src

        return np.array(x_list), np.array(y_list), np.array(distances)

def sample_projmass(probability_density, Nsamples, pixel_scale, x_0, y_0, Rmax, smoothing_scale=4): 
    """
    This function samples a probability density map.

    Parameters
    ----------
    probability_density : TYPE        
    Nsamples : TYPE        
    pixel_scale : TYPE        
    x_0 : TYPE        
    y_0 : TYPE        
    Rmax : radius of rendering area (already scaled) (arcsec) 
    smoothing_scale : Pixel scale over which to smooth. The default is 4.

    Returns
    -------
    coord_x : x-coordinate of point (arcsec)   
    coord_y : y-coordinate of point (arcsec)   
        

    """
    
    # pixel_scale = window size in arcseconds / number of pixels
    # x_0, y_0 = center of window in arcsec
    # Rmax = maximum radius in which to keep points in arcsec
    
    probnorm = probability_density/probability_density.sum()
    
    s = probnorm.shape[0]
    p = probnorm.ravel()

    values = np.arange(s**2)

    # I think we need to sample with replacement, even if point masses end up on top of each other. 
    # Otherwise we might not correctly populate the very high density regions i.e. the central parts of halos
    # And we need to have some overlap if Nsamples > s ** 2 
    
    x_out, y_out = np.array([]), np.array([])

    ndraw = Nsamples
    
    # Put this in since cutting off the corners to maintin circular sym. can remove some halos and the number of
    # objects returned doesn't always equal Nsamples
    while ndraw > 0:
        
        ndraw = Nsamples - len(x_out)
        
        inds=np.random.choice(values,p=p,size=ndraw,replace=True)

        pairs=np.indices(dimensions=(s,s)).T 

        locations = pairs.reshape(-1,2)[inds]
        x_sample_pixel, y_sample_pixel = locations[:,0], locations[:,1]

        # transform to arcsec
        x_sample_arcsec = (x_sample_pixel - s/2) * pixel_scale
        y_sample_arcsec = (y_sample_pixel - s/2) * pixel_scale

        # smooth on sub-pixel scale
        pixel_smoothing_kernel = pixel_scale/smoothing_scale
        # apply smoothing to remove artificial tiling
        x_sample_arcsec += np.random.normal(0, pixel_smoothing_kernel, ndraw)
        y_sample_arcsec += np.random.normal(0, pixel_smoothing_kernel, ndraw)

        # keep circular symmetry
        r = np.sqrt(x_sample_arcsec ** 2 + y_sample_arcsec ** 2)
        keep = np.where(r <= Rmax)
        x_out = np.append(x_out, x_sample_arcsec[keep])
        y_out = np.append(y_out, y_sample_arcsec[keep])
        
        coord_x = x_out + x_0
        coord_y = y_out + y_0
        
    return coord_x, coord_y

def sample_circle(max_rendering_range, Nsmooth, center_x, center_y):
    """
    This function distributes points smoothly accross a plane.

    Parameters
    ----------
    max_rendering_range : radius of rendering area (already scaled) (arcsec) 
    Nsmooth : number of points to render
    center_x : center x coordinate of image
    center_y : center y coordinate of image
        

    Returns
    -------
    coord_x_smooth : x-coordinate of point (arcsec)   
    coord_y_smooth : y-coordinate of point (arcsec) 
        

    """
    # SAMPLE UNIFORM POINTS IN A CIRCLE
    radii = np.random.uniform(0, max_rendering_range ** 2, Nsmooth) 
    # note you have to sample out to r^2 and then take sqrt
    angles = np.random.uniform(0, 2 * np.pi, Nsmooth)
    coord_x_smooth = radii ** 0.5 * np.cos(angles) + center_x
    coord_y_smooth = radii ** 0.5 * np.sin(angles) + center_y
    return coord_x_smooth, coord_y_smooth

def sample_clustered(lens_model_list_at_plane, center_x, center_y, 
             kwargs_lens_at_plane, Nclumpy, max_rendering_range, npix):
    """
    This function distributes points to cluster in areas of higher mass.

    Parameters
    ----------
    lens_model_list_at_plane : model at lensing plane     
    center_x : center x coordinate of image
    center_y : center y coordinate of image
    kwargs_lens_at_plane : arguments from realization instance      
    Nclumpy : number of points to render
    max_rendering_range : radius of rendering area (already scaled) (arcsec)    
    npix : number of pixels on one axis
        

    Returns
    -------
    coord_x_clumpy : x-coordinate of point (arcsec)    
    coord_y_clumpy : y-coordinate of point (arcsec)
        

    """
    grid_x_base = np.linspace(-max_rendering_range, max_rendering_range, npix)
    grid_y_base = np.linspace(-max_rendering_range, max_rendering_range, npix)
    pixel_scale = 2*max_rendering_range/npix
    xx_base, yy_base = np.meshgrid(grid_x_base, grid_y_base)
    shape0 = xx_base.shape
    lens_model_at_plane = LensModel(lens_model_list_at_plane)
    xcoords, ycoords = xx_base + center_x, yy_base + center_y
    projected_mass = lens_model_at_plane.kappa(xcoords.ravel(), ycoords.ravel(), kwargs_lens_at_plane)
    probability_density = projected_mass.reshape(shape0)
    coord_x_clumpy, coord_y_clumpy = sample_projmass(probability_density, Nclumpy, pixel_scale, 
                                           center_x, center_y, max_rendering_range)
    return coord_x_clumpy, coord_y_clumpy