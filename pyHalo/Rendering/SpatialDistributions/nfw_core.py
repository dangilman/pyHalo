import numpy as np
from lenstronomy.LensModel.Profiles.cnfw import CNFW
from pyHalo.Rendering.SpatialDistributions.compute_nfw_fast import FastNFW
import inspect

local_path = inspect.getfile(inspect.currentframe())[0:-11] + 'nfw_tables/'

class ProjectedNFW(object):

    """
    This class approximates sampling from a full 3D NFW profile by
    sampling the projected mass of a cored NFW profile in 2D, and then sampling
    the z coordinate from a cored isothermal profile. This is MUCH faster than sampling from the
    3D NFW profile, and is accurate to within a few percent.
    """

    def __init__(self, rendering_radius, Rs, r_core_host, r200):

        self._cnfw_profile = CNFW()

        self.rmax2d_kpc = rendering_radius
        self._rs_kpc = Rs

        self._xmin = 1e-4
        self.xmax_2d = rendering_radius / Rs

        self.xtidal = r_core_host / Rs
        self.zmax_units_rs = r200 / Rs

        self._xmin = rendering_radius / 30 / self._rs_kpc
        self._norm = self._cnfw_profile._F(self._xmin, self.xtidal)

    @classmethod
    def from_keywords_master(self, keywords_master, lens_cosmo, geometry):

        keywords = self.keywords(keywords_master, lens_cosmo, geometry)

        rendering_radius, Rs, r_core_host, r200 = keywords['rendering_radius'], \
                                                  keywords['Rs'], \
                                                  keywords['r_core'], \
                                                  keywords['host_r200']

        return ProjectedNFW(rendering_radius, Rs, r_core_host, r200)

    @staticmethod
    def keywords(keywords_master, lenscosmo, geometry):

        args_spatial = {}
        kpc_per_arcsec_zlens = geometry.kpc_per_arcsec_zlens
        zlens = lenscosmo.z_lens

        # EVERYTHING EXPRESSED IN KPC
        args_spatial['rendering_radius'] = 0.5 * keywords_master['cone_opening_angle'] * kpc_per_arcsec_zlens

        if 'log_m_host' in keywords_master.keys():
            keywords_master['host_m200'] = 10 ** keywords_master['log_m_host']

        if 'host_m200' in keywords_master.keys():
            # EVERYTHING EXPRESSED IN KPC
            if 'host_c' not in keywords_master.keys():
                keywords_master['host_c'] = lenscosmo.NFW_concentration(keywords_master['host_m200'], zlens,
                                                                        model='diemer19', mdef='200c', logmhm=keywords_master['log_mc'],
                                                                        scatter=True,
                                                                        c_scale=keywords_master['c_scale'], c_power=keywords_master['c_power'],
                                                                        scatter_amplitude=keywords_master['c_scatter_dex'])

            if 'host_Rs' not in keywords_master.keys():
                host_Rs = lenscosmo.NFW_params_physical(keywords_master['host_m200'],
                                                        keywords_master['host_c'], zlens)[1]
                host_r200 = host_Rs * keywords_master['host_c']
            else:
                host_Rs = keywords_master['host_Rs']
                host_r200 = keywords_master['host_Rs'] * keywords_master['host_c']

            args_spatial['Rs'] = host_Rs
            args_spatial['rmax3d'] = host_r200
            args_spatial['host_r200'] = host_Rs * keywords_master['host_c']

        else:
            raise Exception('Must specify the host halo mass when rendering subhalos')

        if 'r_tidal' in keywords_master.keys():

            if isinstance(keywords_master['r_tidal'], str):
                if keywords_master['r_tidal'] == 'Rs':
                    args_spatial['r_core'] = args_spatial['Rs']
                else:
                    if keywords_master['r_tidal'][-2:] != 'Rs':
                        raise ValueError('if specifying the tidal core radius as number*Rs, the last two '
                                         'letters in the string must be "Rs".')

                    scale = float(keywords_master['r_tidal'][:-2])
                    args_spatial['r_core'] = scale * args_spatial['Rs']

            else:
                args_spatial['r_core'] = keywords_master['r_tidal']

        return args_spatial

    def cdf(self, u):

        arg = u * np.arctan(self.zmax_units_rs/self.xtidal)
        return self.xtidal * np.tan(arg)

    def _projected_pdf(self, r2d_kpc):

        x = r2d_kpc / self._rs_kpc

        if isinstance(x, float) or isinstance(x, int):
            x = max(x, self._xmin)
        else:
            x[np.where(x < self._xmin)] = self._xmin

        p = self._cnfw_profile._F(x, self.xtidal) / self._norm

        return p

    def draw(self, N, rescale=1.0, center_x=0., center_y=0.):

        if N == 0:
            return [], [], [], []
        n = 0

        while True:

            _x_kpc, _y_kpc, _r2d, _r3d = self._draw_uniform(N, rescale, center_x, center_y)

            prob = self._projected_pdf(_r2d)
            u = np.random.uniform(size=len(prob))
            keep = np.where(u < prob)[0]

            if n == 0:
                x_kpc = _x_kpc[keep]
                y_kpc = _y_kpc[keep]
                r3d = _r3d[keep]
            else:
                x_kpc = np.append(x_kpc, _x_kpc[keep])
                y_kpc = np.append(y_kpc, _y_kpc[keep])
                r3d = np.append(r3d, _r3d[keep])

            n += len(keep)

            if n >= N:
                break

        return x_kpc[0:N], y_kpc[0:N], r3d[0:N]

    def _draw_uniform(self, N, rescale=1.0, center_x=0., center_y=0.):

        if N == 0:
            return [], [], [], []

        angle = np.random.uniform(0, 2 * np.pi, int(N))

        rmax = self.xmax_2d * rescale

        r = np.random.uniform(0, rmax ** 2, int(N))

        x_arcsec = r ** .5 * np.cos(angle)
        y_arcsec = r ** .5 * np.sin(angle)

        x_arcsec += center_x
        y_arcsec += center_y

        x_kpc, y_kpc = x_arcsec * self._rs_kpc, y_arcsec * self._rs_kpc
        u = np.random.uniform(self._xmin, 0.999999, len(x_kpc))
        z_units_rs = self.cdf(u)
        z_kpc = z_units_rs * self._rs_kpc

        return np.array(x_kpc), np.array(y_kpc), np.hypot(x_kpc, y_kpc), np.sqrt(x_kpc ** 2 + y_kpc**2 + z_kpc ** 2)


class NFW3DFast(object):

    """
    Same as NFW3D, but uses pre-computed CDFs to do the sampling much faster, but still slower than UniformNFW
    """
    def __init__(self, Rs, rmax2d, rmax3d):

        self._Rs = Rs
        self._rmax2d = rmax2d
        self._rmax3d = rmax3d

        self._xc = 0.001 * Rs

        self._xmin = 0.01 * rmax2d / self._Rs

        self._c = rmax3d/Rs

        self.sampler = FastNFW(local_path)

    def _draw(self, N, zlens):

        x, y, z = self.sampler.sample(self._c, N)
        r2 = np.sqrt(x**2 + y**2)
        keep = np.where(r2 <= self._rmax2d/self._Rs)[0]

        return x[keep], y[keep], z[keep]

    def draw(self, N, zlens):

        x, y, z = self._draw(N, zlens)

        while len(x) < N:

            _x, _y, _z = self._draw(N, zlens)
            x = np.append(x, _x)
            y = np.append(y, _y)
            z = np.append(z, _z)

        x_kpc = x[0:N] * self._Rs
        y_kpc = y[0:N] * self._Rs
        z_kpc = z[0:N] * self._Rs
        r2_kpc = np.sqrt(x_kpc ** 2 + y_kpc**2)
        r3_kpc = np.sqrt(r2_kpc ** 2 + z_kpc**2)

        return x_kpc, y_kpc, r3_kpc

class NFW3DCoreRejectionSampling(object):

    """
    Samples from a cored NFW profile with any core radius (within reason) using rejection sampling,
    can be slow.
    """
    def __init__(self, Rs, rmax2d, rmax3d, r_core_parent):

        self._Rs = Rs
        self._rmax2d = rmax2d
        self._rmax3d = rmax3d
        self._x3dmax = rmax3d / Rs

        self._xcore = r_core_parent / Rs

        self.nfw = NFW3DFast(Rs, rmax2d, rmax3d)

        self._xmin = self.nfw._xmin

        self._norm = ((self._xmin + self._xcore) * (1 + self._xmin) ** 2) ** -1

    def _eval_rho_core(self, x, xcore):

        if isinstance(x, int) or isinstance(x, float):
            x = max(self._xmin, x)
        else:
            x[np.where(x < self._xmin)] = self._xmin

        rho = ((x + xcore) * (1 + x) ** 2) ** -1

        return rho/self._norm

    def p_x(self, x, xcore):

        p1 = self._eval_rho_core(x, xcore)
        p2 = self._eval_rho_core(x, 0.)
        return p1/p2

    def _draw(self, N, zlens):

        x, y, r3 = self.nfw.draw(N, zlens)
        x3 = r3/self._Rs
        #prob = np.array([self.p_x(x3i, self._xcore) for x3i in x3])
        prob = np.array(self.p_x(x3, self._xcore))
        u = np.random.rand(N)
        keep = np.where(u < prob)[0]

        return x[keep], y[keep], r3[keep]

    def draw(self, N, zlens):

        x, y, r3 = self._draw(N, zlens)

        while len(x) < N:
            _x, _y, _r3 = self._draw(N, zlens)
            x = np.append(x, _x)
            y = np.append(y, _y)
            r3 = np.append(r3, _r3)



        return x[0:N], y[0:N], r3[0:N]

