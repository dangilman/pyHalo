import numpy as np
from scipy.interpolate import interp2d

zeta_bins = np.array([2,5,10,20,30,40,50])
rho_bins = np.array([5 * 10 ** 6, 10 ** 7, 5 * 10 ** 7, 10 ** 8, 5 * 10 ** 8])
rs_bins = np.array(10 ** np.array([-0.75, -0.4, -0.25, 0.0, 0.25, 0.50]))

rho_array, rs_array = np.meshgrid(rho_bins, rs_bins)
logrhoarray, logrsarray = np.round(np.log10(rho_array).T,2), np.round(np.log10(rs_array).T,2)

interp_tables = []

coords = np.vstack([logrhoarray.ravel(),logrsarray.ravel()]).T

ratio_table = np.array([[[-4.649e+00, -3.845e+00, -3.255e+00, -2.659e+00, -2.326e+00,
         -2.114e+00, -1.954e+00],
        [-3.954e+00, -3.149e+00, -2.602e+00, -2.015e+00, -1.732e+00,
         -1.551e+00, -1.422e+00],
        [-3.645e+00, -2.850e+00, -2.300e+00, -1.778e+00, -1.531e+00,
         -1.358e+00, -1.242e+00],
        [-3.157e+00, -2.377e+00, -1.848e+00, -1.428e+00, -1.255e+00,
         -1.092e+00, -1.000e+00],
        [-2.699e+00, -1.932e+00, -1.480e+00, -1.145e+00, -1.000e+00,
         -8.770e-01, -7.910e-01],
        [-2.204e+00, -1.556e+00, -1.204e+00, -9.130e-01, -7.780e-01,
         -6.950e-01, -6.410e-01]],

       [[-3.746e+00, -2.977e+00, -2.398e+00, -1.875e+00, -1.602e+00,
         -1.423e+00, -1.301e+00],
        [-3.057e+00, -2.301e+00, -1.778e+00, -1.362e+00, -1.176e+00,
         -1.037e+00, -9.590e-01],
        [-2.778e+00, -2.021e+00, -1.593e+00, -1.185e+00, -1.000e+00,
         -9.080e-01, -8.450e-01],
        [-2.301e+00, -1.622e+00, -1.255e+00, -9.540e-01, -8.450e-01,
         -7.230e-01, -6.990e-01],
        [-1.851e+00, -1.301e+00, -9.920e-01, -7.780e-01, -6.410e-01,
         -5.630e-01, -5.170e-01],
        [-1.477e+00, -1.037e+00, -8.330e-01, -6.020e-01, -5.020e-01,
         -4.300e-01, -4.150e-01]],

       [[-1.782e+00, -1.255e+00, -9.630e-01, -7.630e-01, -6.430e-01,
         -5.340e-01, -4.860e-01],
        [-1.301e+00, -9.130e-01, -6.990e-01, -5.110e-01, -4.150e-01,
         -3.580e-01, -3.100e-01],
        [-1.146e+00, -8.020e-01, -6.020e-01, -4.470e-01, -3.800e-01,
         -2.940e-01, -2.560e-01],
        [-9.130e-01, -6.410e-01, -4.770e-01, -3.140e-01, -2.790e-01,
         -2.040e-01, -2.040e-01],
        [-7.320e-01, -4.940e-01, -3.340e-01, -2.300e-01, -1.760e-01,
         -1.460e-01, -1.460e-01],
        [-5.800e-01, -3.540e-01, -2.330e-01, -1.580e-01, -9.200e-02,
         -9.200e-02, -7.100e-02]],

       [[-1.185e+00, -8.390e-01, -6.290e-01, -4.560e-01, -3.770e-01,
         -3.220e-01, -2.860e-01],
        [-8.730e-01, -6.020e-01, -4.620e-01, -3.010e-01, -2.300e-01,
         -2.040e-01, -1.760e-01],
        [-7.600e-01, -5.080e-01, -3.620e-01, -2.550e-01, -1.760e-01,
         -1.460e-01, -1.140e-01],
        [-6.020e-01, -3.980e-01, -2.550e-01, -1.460e-01, -1.020e-01,
         -8.300e-02, -8.600e-02],
        [-4.770e-01, -3.010e-01, -1.670e-01, -1.140e-01, -9.700e-02,
         -7.600e-02, -7.100e-02],
        [-3.420e-01, -1.900e-01, -1.140e-01, -1.000e-01, -8.600e-02,
         -7.600e-02, -7.100e-02]],

       [[-4.470e-01, -2.550e-01, -1.570e-01, -8.200e-02, -9.500e-02,
         -1.500e-01, -2.590e-01],
        [-3.010e-01, -1.760e-01, -9.300e-02, -4.600e-02,  0.000e+00,
          4.000e-03,  1.700e-02],
        [-2.300e-01, -1.100e-01, -1.040e-01, -8.100e-02, -5.100e-02,
         -3.600e-02, -1.300e-02],
        [-1.390e-01, -9.100e-02, -7.600e-02, -4.100e-02, -2.200e-02,
         -1.300e-02,  0.000e+00],
        [-9.200e-02, -4.600e-02, -1.800e-02,  1.700e-02,  4.100e-02,
          6.800e-02,  9.700e-02],
        [-1.140e-01, -8.100e-02, -6.600e-02, -6.000e-02, -5.600e-02,
         -5.600e-02, -7.900e-02]]])

for i in range(0, len(zeta_bins)):
    ratios = ratio_table[:, :, i]
    logratio = []

    for row in range(0, len(rho_bins)):
        for col in range(0, len(rs_bins)):
            logratio.append(ratios[row, col])

    interp_tables.append(interp2d(coords[:, 0], coords[:, 1], logratio))

def do_interp(lrho, lrs, lzeta):
    if isinstance(lzeta, float) or isinstance(lzeta, int):
        return do_interp_single(lrho, lrs, lzeta)[0]
    else:
        out = []
        for lzi in lzeta:
            out.append(do_interp_single(lrho, lrs, lzi))
        return np.array(out)


def do_interp_single(lrho, lrs, lzeta):
    lzbins = np.round(np.log10(zeta_bins), 2)

    sorted_inds = np.argsort(np.absolute(lzeta - lzbins))

    if lzeta <= lzbins[-1] and lzeta >= lzbins[0]:

        closest_ind = sorted_inds[0]
        second_closest_ind = sorted_inds[1]

        closest_zeta = lzbins[closest_ind]
        second_closest_zeta = lzbins[second_closest_ind]

        diff = np.absolute(second_closest_zeta - closest_zeta)

        weight1 = 1 - np.absolute(lzeta - closest_zeta) * diff ** -1
        weight2 = 1 - weight1
        # print(weight1, weight2)
        f1, f2 = interp_tables[closest_ind](lrho, lrs), interp_tables[second_closest_ind](lrho, lrs)
        return 10 ** (weight1 * f1 + weight2 * f2)

    elif lzeta > lzbins[-1]:

        last_zeta_midpoint = lzbins[-3]

        log_value_at_midpoint = np.log10(do_interp_single(lrho, lrs, last_zeta_midpoint))

        d_log_zeta = lzbins[-1] - last_zeta_midpoint

        derivative = (interp_tables[-1](lrho, lrs) - log_value_at_midpoint) * d_log_zeta ** -1

        intercept = interp_tables[-1](lrho, lrs)

        delta_zeta = lzeta - lzbins[-1]

        return 10 ** (derivative * delta_zeta + intercept)

    else:

        return do_interp_single(lrho, lrs, lzbins[0])

def halo_age(z, lookback_time_function, zform):

    formation_time = lookback_time_function(zform)
    age = lookback_time_function(z) - formation_time
    return np.max(age, 0)

def zeta_value(z, cross_0, lookback_time_function, zform):
    age = halo_age(z, lookback_time_function, zform)
    return age * cross_0

