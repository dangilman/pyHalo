import numpy as np

v_power = 0.5

zeta_values = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
delta_zeta = 10

slopes_fit5 = np.array([ 0.0012989 , -0.00151646, -0.15982686])
intercept_fit5 = np.array([-0.01229278,  0.0428837 ,  9.49031537])

slopes_fit10 = np.array([-0.00025443,  0.01007318, -0.17423818])
intercept_fit10 = np.array([-0.0169306 ,  0.02538956,  9.363078  ])

slopes_fit20 = np.array([ 0.00320873, -0.00437765, -0.16328214])
intercept_fit20 = np.array([-0.05463304,  0.18431916,  9.08253523])

slopes_fit30 = np.array([ 0.00308802, -0.00220246, -0.16433341])
intercept_fit30 = np.array([-0.05864577,  0.19309639,  8.98506665])

slopes_fit40 = np.array([ 0.00321089, -0.00261928, -0.16619568])
intercept_fit40 = np.array([-0.06121329,  0.20783084,  8.93435083])

slopes_fit50 = np.array([ 0.00248163,  0.00078203, -0.16693899])
intercept_fit50 = np.array([-0.05739456,  0.19123372,  8.89263283])

slopes_fit60 = np.array([ 0.00483479, -0.0057663 , -0.16375015])
intercept_fit60 = np.array([-0.07757196,  0.25334904,  8.82731654])

slopes_fit70 = np.array([ 0.00465157, -0.00559884, -0.16428474])
intercept_fit70 = np.array([-0.07582748,  0.2570401 ,  8.80085341])

slopes_fit80 = np.array([ 0.00353611, -0.00084991, -0.16869627])
intercept_fit80 = np.array([-0.06987584,  0.23252822,  8.80845793])

slopes_fit90 = np.array([ 0.00476609, -0.00431403, -0.16640688])
intercept_fit90 = np.array([-0.08219107,  0.27246728,  8.76596329])

slopes_fit100 = np.array([ 0.0046619 , -0.00344269, -0.1671239 ])
intercept_fit100 = np.array([-0.08314166,  0.27455759,  8.75212681])

slopes_fit110 = np.array([ 3.60339922e-03,  1.31360162e-04, -1.69504826e-01])
intercept_fit110 = np.array([-0.07480173,  0.24838737,  8.7579539 ])

slopes_fit120 = np.array([ 0.00449002, -0.00314914, -0.16767864])
intercept_fit120 = np.array([-0.084442  ,  0.28699014,  8.72455021])

slope_polynomial = [slopes_fit5 ,slopes_fit10, slopes_fit20, slopes_fit30, slopes_fit40, slopes_fit50, slopes_fit60,
          slopes_fit70, slopes_fit80, slopes_fit90, slopes_fit100, slopes_fit110, slopes_fit120]

intercept_polynomial = [intercept_fit5, intercept_fit10, intercept_fit20, intercept_fit30, intercept_fit40, intercept_fit50,
              intercept_fit60, intercept_fit70, intercept_fit80, intercept_fit90,
              intercept_fit100, intercept_fit110, intercept_fit120]

def _logrho_Mz(m, z, idx):

    slope = slope_polynomial[idx][0] * z ** 2 + slope_polynomial[idx][1] * z + slope_polynomial[idx][2]
    intercept = intercept_polynomial[idx][0] * z ** 2 + intercept_polynomial[idx][1] * z + intercept_polynomial[idx][2]
    # print(slope, intercept)

    return intercept + np.log10(m) * slope

def logrho_Mz_5(m, z, zeta, cmean, c):

    if zeta >= 5 and zeta < 10:

        inds = np.argsort(np.absolute(zeta_values - zeta))[0:2]

        w1 = np.absolute(1 - np.absolute(zeta - zeta_values[inds[0]]) * 5 ** -1)
        w2 = 1 - w1
        # w1, w2 = 1, 0

        rho1 = w1 * _logrho_Mz(m, z, inds[0])
        rho2 = w2 * _logrho_Mz(m, z, inds[1])

        rho_central = rho1 + rho2

    elif zeta >= 10 and zeta < 12.5:
        inds = [1, 2]

        w1 = np.absolute(1 - np.absolute(zeta - zeta_values[inds[0]]) * delta_zeta ** -1)
        w2 = 1 - w1
        # w1, w2 = 1, 0

        rho1 = w1 * _logrho_Mz(m, z, inds[0])
        rho2 = w2 * _logrho_Mz(m, z, inds[1])

        rho_central = rho1 + rho2

    elif zeta < zeta_values[0]:
        rho_0 = _logrho_Mz(m, z, 0)

        rho_at_0 = 10

        derivative = (rho_0 - rho_at_0) * delta_zeta ** -1

        rho_central = (zeta - zeta_values[0]) * derivative + rho_0

    elif zeta > 120:
        nmax = int(len(zeta_values))-1
        rho_0 = _logrho_Mz(m, z, nmax)
        derivative = (-_logrho_Mz(m, z, int(nmax-1)) + rho_0) * delta_zeta ** -1
        rho_central = (zeta - zeta_values[-1]) * derivative + rho_0

    else:

        inds = np.argsort(np.absolute(zeta_values - zeta))[0:2]

        w1 = np.absolute(1 - np.absolute(zeta - zeta_values[inds[0]]) * delta_zeta ** -1)
        w2 = 1 - w1
        #w1, w2 = 1, 0

        rho1 = w1 * _logrho_Mz(m, z, inds[0])
        rho2 = w2 * _logrho_Mz(m, z, inds[1])

        rho_central = rho1 + rho2

    delta_c = c * cmean ** -1 - 1
    slope_c = 0.5

    rho_central = 10**rho_central * (1 - slope_c * delta_c)

    return np.log10(rho_central)
