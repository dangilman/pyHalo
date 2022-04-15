import numpy as np
from pyHalo.defaults import lenscone_default
from scipy.interpolate import interp1d
from pyHalo.Cosmology.cosmology import Cosmology
from scipy.integrate import quad
from pyHalo.Halos.lens_cosmo import LensCosmo
from scipy.special import jv
from scipy.integrate import simps


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

def sample_density(probability_density, Nsamples, pixel_scale, x_0, y_0, Rmax, smoothing_scale=4):
    """

    :param probability_density:
    :param Nsamples:
    :param pixel_scale:
    :param x_0:
    :param y_0:
    :param Rmax:
    :param smoothing_scale:
    :return:
    """

    probnorm = probability_density / probability_density.sum()

    s = probnorm.shape[0]
    p = probnorm.ravel()

    values = np.arange(s ** 2)

    x_out, y_out = np.array([]), np.array([])

    ndraw = Nsamples

    while ndraw > 0:
        ndraw = Nsamples - len(x_out)

        inds = np.random.choice(values, p=p, size=ndraw, replace=True)

        pairs = np.indices(dimensions=(s, s)).T

        locations = pairs.reshape(-1, 2)[inds]
        x_sample_pixel, y_sample_pixel = locations[:, 0], locations[:, 1]

        # transform to arcsec
        x_sample_arcsec = (x_sample_pixel - s / 2) * pixel_scale
        y_sample_arcsec = (y_sample_pixel - s / 2) * pixel_scale

        # smooth on sub-pixel scale
        pixel_smoothing_kernel = pixel_scale / smoothing_scale
        # apply smoothing to remove artificial tiling
        x_sample_arcsec += np.random.normal(0, pixel_smoothing_kernel, ndraw)
        y_sample_arcsec += np.random.normal(0, pixel_smoothing_kernel, ndraw)

        # keep circular symmetry
        r = np.sqrt(x_sample_arcsec ** 2 + y_sample_arcsec ** 2)
        keep = np.where(r <= Rmax)
        x_out = np.append(x_out, x_sample_arcsec[keep])
        y_out = np.append(y_out, y_sample_arcsec[keep])

    # originally this returned coord_x and coord_y, shouldn't it return x_out and y_out?
    return x_out, y_out

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

def sample_clustered(lens_model, kwargs_lens, center_x, center_y, n_samples, max_rendering_range, npix):
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
    pixel_scale = 2 * max_rendering_range / npix
    xx_base, yy_base = np.meshgrid(grid_x_base, grid_y_base)
    shape0 = xx_base.shape

    xcoords, ycoords = xx_base + center_x, yy_base + center_y
    projected_mass = lens_model.kappa(xcoords.ravel() + center_x, ycoords.ravel() + center_y, kwargs_lens).reshape(shape0)
    coord_x, coord_y = sample_density(projected_mass, n_samples, pixel_scale,
                                                     center_x, center_y, max_rendering_range)
    return coord_x, coord_y

def de_broglie_wavelength(log10_m_uldm,v):
    '''
    Returns de Broglie wavelength of the ultra-light axion in kpc.

    :param log10_m_uldm: log(axion mass) in eV
    :param v: velocity in km/s
    '''
    m_axion=10**log10_m_uldm
    return 1.2*(1e-22/m_axion)*(100/v)

def delta_kappa(z_lens, z_source, m, rein, de_Broglie_wavelength):
    '''
    Returns standard deviation of the density fluctuations in projection in convergence units

    :param z_lens,z_source: lens and source redshifts
    :param m: main deflector halo mass in M_solar
    :param rein: Einstein radius in kpc
    :param de_Broglie_wavelength: de Broglie wavelength of axion in kpc
    '''
    l = LensCosmo(z_lens,z_source)
    sigma_crit = l.get_sigma_crit_lensing(z_lens, z_source) * (1e-3) ** 2
    ds = delta_sigma(m, z_lens, rein, de_Broglie_wavelength)
    delta_kappa = ds / sigma_crit

    return delta_kappa

def delta_sigmaNFW(z_lens, m, rein, de_Broglie_wavelength):
    '''
    Returns standard deviation of the density fluctuations in projection normalized by the projected
    density of the host halo

    :param z_lens,z_source: lens and source redshifts
    :param m: main deflector halo mass in M_solar
    :param rein: Einstein radius in kpc
    :param de_Broglie_wavelength: de Broglie wavelength of axion in kpc
    '''

    l = LensCosmo(None, None)
    c = l.NFW_concentration(m, z_lens, scatter=False)
    rhos, rs, _ = l.NFW_params_physical(m, c, z_lens)
    kappa_host = projected_squared_density(rein, rhos, rs, c) ** 0.5
    ds = delta_sigma(m, z_lens, rein, de_Broglie_wavelength)
    return ds/kappa_host

def delta_sigma(m, z, rein, de_Broglie_wavelength):
    """
    Returns the mean ULDM fluctuation ampltiude of the host dark matter halo in units M_sun/kpc^2
    :param m:
    :param z:
    :param rein:
    :param de_Broglie_wavelength:
    :return:
    """
    l = LensCosmo(None, None)
    c = l.NFW_concentration(m, z, scatter=False)
    rhos, rs, _ = l.NFW_params_physical(m, c, z)
    nfw_rho_squared = projected_density_squared(rein, rhos, rs, c)
    delta_sigma = (np.sqrt(np.pi) * nfw_rho_squared * de_Broglie_wavelength)**0.5
    return delta_sigma

def delta_sigma_kawai(r, mhost, zhost, lambda_dB, dm_density_over_stellar_density):
    """

    :param lambda_dB:
    :param effective_halo_size:
    :param baryon_fraction:
    :return:
    """

    l = LensCosmo()
    c = l.NFW_concentration(mhost, zhost, scatter=False)
    rhos, rs, _ = l.NFW_params_physical(mhost, c, zhost)
    reff = effective_halo_size(r, rhos, rs, c)
    f = dm_density_over_stellar_density / (dm_density_over_stellar_density + 1)

    window_function = lambda x: jv(1, x)/x # FFT of circular tophat
    dB_volume = 4 * np.pi * lambda_dB / 3
    integrand = lambda k: 2 * np.pi * k * np.exp(-0.25 * k ** 2 * lambda_dB ** 2) * window_function(lambda_dB * k) ** 2
    integral = quad(integrand, 0, 100 * lambda_dB)[0] / (2 * np.pi ** 2)  # has units length^-2
    prefactor = f ** 2 * dB_volume / reff  # has units length^2
    return np.sqrt(prefactor * integral)

def nfwF(x):
    if x < 1:
        return np.arctanh(np.sqrt(1-x**2))/(np.sqrt(1-x**2))
    else:
        return np.arctan(np.sqrt(x ** 2 - 1)) / (np.sqrt(x ** 2 - 1))

def projected_density_squared(R_ein, rhos, rs, concentration):

    '''
    Returns the integral along the line of sight of the square of the NFW profile
    i.e. integral rho_nfw(z)^2 dz

    :param R_ein: Einstein radius in kpc
    :param rhos: scale radius density of main deflector halo
    :param rs: scale radius of main deflector halo
    :param concentration: concentration of main deflector halo
    '''

    r200 = concentration * rs
    zmax = np.sqrt(r200 ** 2 - R_ein**2)

    x = lambda z: np.sqrt(R_ein ** 2 + z ** 2)/rs
    nfw_density_square = lambda z: rhos**2 / (x(z) * (1+x(z))**2)**2

    return 2 * quad(nfw_density_square, 0, zmax)[0]

def projected_squared_density(R_ein, rhos, rs, concentration):
    '''
    Returns integral along the line of sight of the NFW profile squared
    i.e. (integral rho_nfw(z) dz)^2

    :param R_ein: Einstein radius in kpc
    :param rhos: scale radius density of main deflector halo
    :param rs: scale radius of main deflector halo
    :param concentration: concentration of main deflector halo
    '''

    r200 = concentration * rs
    zmax = np.sqrt(r200 ** 2 - R_ein ** 2)

    x = lambda z: np.sqrt(R_ein ** 2 + z ** 2) / rs
    nfw_density = lambda z: rhos / (x(z) * (1 + x(z)) ** 2)
    integral = 2 * quad(nfw_density, 0, zmax)[0]
    return integral ** 2

def effective_halo_size(r, rhos, rs, concentration):
    """
    Computes the effective halo size as defined in Equation 17 of Kawai et al. (2021)
    :param r: the radius at which to evaluate the size
    [can be any unit of length or an angle, as long the definition is
    consistent also between rhos, and rs]
    :param rhos: the density parameter the host NFW halo [units M_sun / length^3]
    :param rs: the scale radius of host halo [in the same units as r]
    :param concentration: host halo concentration
    :return: the effective halo size in the same units as r
    """
    denom = projected_density_squared(r, rhos, rs, concentration)
    num = projected_squared_density(r, rhos, rs, concentration)
    return num/denom


def nfw_velocity_dispersion(rhos, rs, c, x=1):
    """
    Computes the central velocity dispersion of an NFW profile integrating from r = x * rs
    :param rhos:
    :param rs:
    :param c:
    :return:
    """
    G = 4.3e-6  # kpc/solMass * (km/sec)**2
    prefactor = 4 * np.pi * G * rhos ** 2 * rs ** 2
    density_at_r = rhos / (x * (1 + x) ** 2)
    _integrand = lambda x: (np.log(1 + x) - x / (1 + x)) / (x * (1 + x) ** 2) / x ** 2

    x = np.linspace(x, c, 150)
    y = _integrand(x)
    return np.sqrt(prefactor * simps(y, x) / density_at_r)

def nfw_velocity_dispersion_fromfit(m):
    """
    The velocity dispersion of an NFW profile with mass m calibrated from a power law fit for halos
    between 10^6 and 10^10 at z=0
    :param m: halo mass in M_sun
    :return: the velocity dispersion inside rs
    """
    coeffs = [0.31575757, -1.74259129]
    log_vrms = coeffs[0] * np.log10(m) + coeffs[1]
    return 10 ** log_vrms
