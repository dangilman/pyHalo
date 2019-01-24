import numpy as np

class NFW_3D(object):

    def __init__(self, Rs, rmax2d, rmax3d, xoffset=0, yoffset = 0, tidal_core=False, r_core = None):

        """
        all distances expressed in (physical) kpc

        :param Rs: scale radius
        :param rmax2d: maximum 2d radius
        :param rmax3d: maximum 3d radius (basically sets the distribution of z coordinates)
        :param xoffset: centroid of NFW
        :param yoffset: centroid of NFW
        :param tidal_core: flag to draw from a uniform denity inside r_core
        :param r_core: format 'number * Rs' where number is a float.
        specifies an inner radius where the distribution is uniform
        see Figure 4 in Jiang+van den Bosch 2016
        """

        self.rmax3d = rmax3d
        self.rmax2d = rmax2d
        self.rs = Rs

        self.xoffset = xoffset
        self.yoffset = yoffset

        rmin = Rs*0.001

        self.xoffset,self.yoffset = xoffset,yoffset
        self.tidal_core = tidal_core
        self.core_fac = 1
        self.r_core = r_core

        self.xmin = rmin * Rs ** -1
        self.xmax = rmax3d * Rs ** -1

    def draw(self,N):

        r3d, x, y, r2d,z = [], [], [], [],[]

        while len(r3d) < N:

            theta = np.random.uniform(0,2*np.pi)
            phi = np.random.uniform(0,2*np.pi)

            r_2 = np.random.uniform(0,self.rmax2d**2) ** 0.5
            r_z = np.random.uniform(0,self.rmax3d**2) ** 0.5

            x_value,y_value = r_2*np.cos(theta),r_2*np.sin(theta)
            z_value = r_z * np.sin(phi)

            r2 = (x_value**2+y_value**2)**0.5

            r3 = (r2**2+z_value**2)**0.5

            if r3*self.rs**-1 <= self.xmin:
                choose = 1

            else:

                if self.tidal_core:
                    choose = self._density_3d(max(self.r_core, r3)) * self._upper_bound(r3) ** -1
                else:
                    choose = self._density_3d(r3) * self._upper_bound(r3) ** -1

            if choose > np.random.uniform(0,1) and r2 <= self.rmax2d:
                r3d.append(r3)
                x.append(x_value+self.xoffset)
                y.append(y_value+self.yoffset)
                z.append(z_value)
                r2d.append(r2)

        x = np.array(x)
        y = np.array(y)
        r2d = np.array(r2d)
        r3d = np.array(r3d)

        return x,y,r2d,r3d

    def _density_3d(self, r):

        x = r*self.rs**-1

        if isinstance(x,float) or isinstance(x,int):
            x = max(self.xmin,x)
        else:
            x[np.where(x<self.xmin)] = self.xmin

        return (x*(1+x)**2)**-1

    def _upper_bound(self, r, alpha=0.9999999):

        X = r*self.rs**-1
        norm = self._density_3d(self.xmin * self.rs)

        if isinstance(X,int) or isinstance(X,float):

            if X>self.xmin:
                return norm*(X*self.xmin**-1)**-alpha
            else:
                return norm

        else:
            X[np.where(X < self.xmin)] = self.xmin
            return norm*(X*self.xmin**-1)**-alpha
