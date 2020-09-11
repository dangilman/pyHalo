import numpy as np
import dill as pickle
import inspect

local_path = inspect.getfile(inspect.currentframe())[0:-6]

class NFW3D(object):

    def __init__(self, Rs, rmax2d, rmax3d):

        self._Rs = Rs
        self._rmax2d = rmax2d
        self._rmax3d = rmax3d
        self._x3dmax = rmax3d / Rs

        self._xmin = 0.01 * rmax2d/self._Rs

        self._norm = (self._xmin * (1 + self._xmin) ** 2) ** -1

        self._x2dmax = self._rmax2d / Rs

    def _eval_rho(self, x):

        x = max(self._xmin, x)
        rho = (x * (1 + x) ** 2) ** -1

        return rho/self._norm

    def _sample_uniform_xy(self):
        theta = np.random.uniform(0, 2 * np.pi)
        x2d = np.sqrt(np.random.uniform(0, self._rmax2d ** 2))
        x_kpc, y_kpc = x2d * np.cos(theta), x2d * np.sin(theta)
        return x_kpc, y_kpc

    def _generate_proposal(self):

        x_kpc, y_kpc = self._sample_uniform_xy()

        r2d = (x_kpc ** 2 + y_kpc ** 2)**0.5
        z_kpc = self._sample_z(r2d)

        return x_kpc, y_kpc, z_kpc

    def _sample_z(self, r2d):

        zmax = np.sqrt(self._x3dmax**2 * self._Rs**2 - r2d**2)
        z = np.random.uniform(0., zmax)
        return z

    def _draw_single(self):

        # all in kpc
        while True:
            xprop, yprop, zprop = self._generate_proposal()

            r2dprop = (xprop ** 2 + yprop ** 2) ** 0.5

            r3dprop = (r2dprop ** 2 + zprop ** 2) ** 0.5

            X3d = r3dprop / self._Rs

            u = np.random.rand()

            prob = self._eval_rho(X3d)

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

        filename = local_path + 'NFWlookup'
        file = open(filename, 'rb')
        self.sampler = pickle.load(file)
        file.close()

    def _draw(self, N, zlens):

        x, y, z = self.sampler.sample(self._c, N)
        r2 = np.sqrt(x**2 + y**2)
        keep = np.where(r2 <= self._rmax2d/self._Rs)[0]

        return x[keep], y[keep], z[keep]

    def draw(self, N, zlens):

        x, y, z = self._draw(N, zlens)

        while len(x) < N:
            _x, _y, _z = self._draw(N - len(x), zlens)
            x = np.append(x, _x)
            y = np.append(y, _y)
            z = np.append(z, _z)

        x_kpc = x * self._Rs
        y_kpc = y * self._Rs
        z_kpc = z * self._Rs
        r2_kpc = np.sqrt(x_kpc ** 2 + y_kpc**2)
        r3_kpc = np.sqrt(r2_kpc ** 2 + z_kpc**2)

        return x_kpc, y_kpc, r2_kpc, r3_kpc
