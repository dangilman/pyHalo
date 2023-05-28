import numpy as np
from pyHalo.defaults import lenscone_default
from scipy.interpolate import interp1d
from pyHalo.Cosmology.cosmology import Cosmology
from scipy.integrate import quad
from pyHalo.Halos.lens_cosmo import LensCosmo
from scipy.integrate import simps
from pyHalo.concentration_models import preset_concentration_models

class ITSampling(object):

    def __init__(self, x, cdf):
        """
        This class performs inverse transform sampling of a distribution given samples from the distribution
        :param samples: samples from the distribution
        """
        cdf = cdf / float(np.max(cdf))
        self._cdf_inverse = interp1d(cdf, x)
        self._umin = cdf[0]
        self._umax = cdf[-1]

    @classmethod
    def from_samples(cls, samples):
        """
        samples from the target distribution
        :param samples:
        :return:
        """
        ran = (np.min(samples), np.max(samples))
        h, x = np.histogram(samples, range=ran, bins=150)
        x = x[0:-1] + (x[1] - x[0]) / 2
        cdf = np.cumsum(h)
        cdf = cdf / float(np.max(cdf))
        return ITSampling(x, cdf)

    def __call__(self, n_samples):
        """
        Generates samples from the distribution
        :param n_samples: number of samples to draw
        :return: the samples
        """
        u = np.random.uniform(self._umin, self._umax, int(n_samples))
        samples_out = self._cdf_inverse(u)
        if n_samples == 1:
            return float(samples_out)
        else:
            return np.squeeze(samples_out)

def inverse_transform_sampling(x, function, args, n_samples):
    """

    :param x: the domain of the function across which you want to obtain samples
    :param function: the function or probability density you want to sample from
    :param args: arguments passed to function after x
    :param n_samples: number of samples to draw
    :return: samples from the probability density described by function
    """
    y = function(x, *args)
    cdf = np.cumsum(y)
    return inverse_transform_sampling_from_cdf(x, cdf, n_samples)

def inverse_transform_sampling_from_cdf(x, cdf, n_samples):
    """

    :param x: the domain of the function across which you want to obtain samples
    :param cdf: the cumulative distribution function of the pdf
    :param n_samples: number of samples to draw
    :return: samples from the probability density that corresponds to cdf
    """
    cdf = cdf * float(np.max(cdf)) ** -1
    cdf_inverse = interp1d(cdf, x)
    u = np.random.uniform(cdf[0], cdf[-1], n_samples)
    return cdf_inverse(u)

def generate_lens_plane_redshifts(zlens, zsource):
    """
    This routine sets up the redshift planes along the line of sight in the lens system
    :param zlens: main deflector plane redshift (if there is no main lens plane, then this can be set as None)
    :param zsource: source plane redshift
    :return: lens plane redshifts and the thickness of each slice
    """
    zmin = lenscone_default.default_zstart
    zstep = lenscone_default.default_z_step
    if zlens is None:
        redshifts = np.arange(zmin, zsource, zstep)
    else:
        front_z = np.arange(zmin, zlens, zstep)
        back_z = np.arange(zlens, zsource, zstep)
        redshifts = np.append(front_z, back_z)
    delta_zs = []
    for i in range(0, len(redshifts) - 1):
        delta_zs.append(redshifts[i + 1] - redshifts[i])
    delta_zs.append(zsource - redshifts[-1])
    return list(np.round(redshifts, 2)), np.round(delta_zs, 2)

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

    cosmo = Cosmology()
    l = LensCosmo(z_lens, 2.0, cosmo)
    model, _ = preset_concentration_models('DIEMERJOYCE19')
    cmodel = model(cosmo.astropy, scatter=False)
    c = cmodel.nfw_concentration(m, z_lens)
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
    cosmo = Cosmology()
    l = LensCosmo(z, 2.0, cosmo)
    model, _ = preset_concentration_models('DIEMERJOYCE19')
    cmodel = model(cosmo.astropy, scatter=False)
    c = cmodel.nfw_concentration(m, z)
    rhos, rs, _ = l.NFW_params_physical(m, c, z)
    nfw_rho_squared = projected_density_squared(rein, rhos, rs, c)
    delta_sigma = (np.sqrt(np.pi) * nfw_rho_squared * de_Broglie_wavelength)**0.5
    return delta_sigma

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

class MinHaloMassULDM(object):

    def __init__(self, log10_m_uldm, astropy_instance, log_mlow):
        """
        This class implements the minimum halo mass for ultra-light dark matter, given the particle mass and a cosmology

        The call method returns the maximum of log_mlow, and the resulting minimum ULDM halo mass to ensure that one does
        not inadvertently generate halos down to extremely low masses for heavy particles
        :param log10_m_uldm: particle mass
        :param astropy_instance: an instannce of astropy
        :param log_mlow: minimum halo mass to render, regardless of ULDM computation
        """
        m22 = 10**(log10_m_uldm + 22)
        self._Mmin0 = 4.4e7 * m22**(-3/2)
        self._astropy_instance = astropy_instance
        self._log_mlow = log_mlow

    def __call__(self, z):
        m_min = self.m_min(z)
        log10_m_min = np.log10(m_min)
        return max(log10_m_min, self._log_mlow)

    def m_min(self, z):
        return self._a(z) ** (-3 / 4) * (self._zeta(z) / self._zeta(0)) ** (1 / 4) * self._Mmin0

    def _a(self, z):
        return (1 + z) ** -1

    def _Om(self, z):
        return self._astropy_instance.Om(z)

    def _zeta(self, z):
        return (18 * np.pi ** 2 + 82 * (self._Om(z) - 1) - 39 * (self._Om(z) - 1) ** 2) / self._Om(z)
