import numpy as np
from scipy.interpolate import interp1d
import dill
import inspect
import os

local_path = inspect.getfile(inspect.currentframe())[0:-6]

class NFW3D(object):

    def __init__(self, Rs, rmax2d, rmax3d, r_core_parent=None):

        self._Rs = Rs
        self._rmax2d = rmax2d
        self._rmax3d = rmax3d
        self._x3dmax = rmax3d / Rs

        if r_core_parent is not None:
            self._xmin = r_core_parent / Rs
        else:
            self._xmin = 0.001 * Rs

        self._norm = self._eval_rho(self._xmin)
        self._x2dmax = self._rmax2d / Rs

    @staticmethod
    def _eval_rho(x):
        return (x * (1 + x) ** 2) ** -1

    def _sample_uniform_xy(self):
        theta = np.random.uniform(0, 2 * np.pi)
        x2d = np.sqrt(np.random.uniform(0, self._rmax2d ** 2))
        x_kpc, y_kpc = x2d * np.cos(theta), x2d * np.sin(theta)
        return x_kpc, y_kpc

    def _generate_proposal(self):

        x_kpc, y_kpc = self._sample_uniform_xy()

        z_kpc = self._sample_z()

        return x_kpc, y_kpc, z_kpc

    def _sample_z(self):
        
        z = np.random.uniform(self._xmin, self._x3dmax * self._Rs)
        return z

    def _density(self, x):

        constant_prob = 1
        max_x = max(self._xmin, x)
        ratio = self._eval_rho(max_x) / self._eval_rho(self._xmin)
        return constant_prob * ratio

    def _draw_single(self):

        # all in kpc
        while True:
            xprop, yprop, zprop = self._generate_proposal()

            r2dprop = (xprop ** 2 + yprop ** 2) ** 0.5

            r3dprop = (r2dprop ** 2 + zprop ** 2) ** 0.5

            X3d = r3dprop / self._Rs

            u = np.random.rand()

            prob = self._density(X3d)

            if prob >= u:
                return xprop, yprop, r2dprop, r3dprop

    def draw(self, N, zlens):

        x_kpc, y_kpc, r2d_kpc, r3d_kpc = [], [], [], []

        for i in range(0, N):
            out = self._draw_single()
            x_kpc.append(out[0])
            y_kpc.append(out[1])
            r2d_kpc.append(out[2])
            r3d_kpc.append(out[3])

        return np.array(x_kpc), np.array(y_kpc), np.array(r2d_kpc), np.array(r3d_kpc)


def approx_cdf_1d(x_array, pdf_array):
    """

    :param x_array: x-values of pdf
    :param pdf_array: pdf array of given x-values
    """
    norm_pdf = pdf_array / np.sum(pdf_array)
    cdf_array = np.zeros_like(norm_pdf)
    cdf_array[0] = norm_pdf[0]
    for i in range(1, len(norm_pdf)):
        cdf_array[i] = cdf_array[i - 1] + norm_pdf[i]
    cdf_func = interp1d(x_array, cdf_array)
    cdf_inv_func = interp1d(cdf_array, x_array)
    return cdf_array, cdf_func, cdf_inv_func


def rhonfw_x(x, norm=1):
    return norm * (x * (1 + x) ** 2) ** -1


def _rhonfw_tidal(x, xtidal):
    norm = rhonfw_x(xtidal)
    if x >= xtidal:
        return rhonfw_x(x)
    else:
        return norm


def rhonfw_tidal(x, xtidal):
    if isinstance(x, float) or isinstance(x, int):
        return _rhonfw_tidal(x, xtidal)
    else:
        vals = []
        for xi in x:
            vals.append(_rhonfw_tidal(xi, xtidal))
        return np.array(vals)

class NFW3DFast(object):

    def __init__(self, Rs, rmax2d, rmax3d, r_core_parent=None):

        self._Rs = Rs
        self._rmax2d = rmax2d
        self._rmax3d = rmax3d

        if r_core_parent is not None:
            self._xc = r_core_parent / Rs
        else:
            self._xc = 0.001 * Rs

        self._c = rmax3d/Rs

        filename = local_path + 'NFWlookup'
        file = open(filename, 'rb')
        self.sampler = dill.load(file)

    def draw(self, N, zlens):

        x_kpc, y_kpc, z_kpc, r2d_kpc = [], [], [], []
        while len(x_kpc) < N:
            _x, _y, _z = self.sampler.sample(self._xc, self._c, 1)
            _x_kpc = _x * self._Rs
            _y_kpc = _y * self._Rs
            _z_kpc = _z * self._Rs
            _r2d_kpc = np.sqrt(_x_kpc**2 + _y_kpc**2)
            if _r2d_kpc <= self._rmax2d:
                x_kpc.append(_x_kpc)
                y_kpc.append(_y_kpc)
                z_kpc.append(_z_kpc)
                r2d_kpc.append(_r2d_kpc)

        x_kpc, y_kpc, z_kpc = np.array(x_kpc), np.array(y_kpc), np.array(z_kpc)
        r2d_kpc = np.array(r2d_kpc)
        r3d_kpc = np.sqrt(z_kpc**2 + r2d_kpc**2)

        return x_kpc, y_kpc, r2d_kpc, r3d_kpc


