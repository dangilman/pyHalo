import numpy as np
from pyHalo.Spatial.compute_nfw_fast import FastNFW
import inspect

local_path = inspect.getfile(inspect.currentframe())[0:-11] + 'nfw_tables/'

class NFW3DFast(object):

    """
    Same as NFW3D, but uses pre-computed CDFs to do the sampling much faster
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

        return x_kpc, y_kpc, r2_kpc, r3_kpc


class NFW3DCore(object):

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

        x, y, r2, r3 = self.nfw.draw(N, zlens)
        x3 = r3/self._Rs
        #prob = np.array([self.p_x(x3i, self._xcore) for x3i in x3])
        prob = np.array(self.p_x(x3, self._xcore))
        u = np.random.rand(N)
        keep = np.where(u < prob)[0]

        return x[keep], y[keep], r2[keep], r3[keep]

    def draw(self, N, zlens):

        x, y, r2, r3 = self._draw(N, zlens)

        while len(x) < N:
            _x, _y, _r2, _r3 = self._draw(N, zlens)
            x = np.append(x, _x)
            y = np.append(y, _y)
            r2 = np.append(r2, _r2)
            r3 = np.append(r3, _r3)



        return x[0:N], y[0:N], r2[0:N], r3[0:N]


