import numpy as np

def render_los(rendering_class, lens_plane_redshifts, delta_zs, zmin, zmax):

    redshifts = []

    init = True

    for i, (zi, delta_zi) in enumerate(zip(lens_plane_redshifts, delta_zs)):

        if zi < zmin or zi > zmax:
            continue

        m = rendering_class.render_masses(zi, delta_zi, None)

        rescale_angle = rendering_class.rescale_angle(zi)
        x, y, r2, r3 = rendering_class.render_positions_at_z(zi, len(m), rescale_angle,
                                                             0., 0.)

        redshifts += [zi] * len(x)

        if init:
            masses = m
            x_arcsec, y_arcsec, r2d, r3d = x, y, r2, r3
            init = False
        else:
            x_arcsec = np.append(x_arcsec, x)
            y_arcsec = np.append(y_arcsec, y)
            masses = np.append(masses, m)
            r2d = np.append(r2d, r2)
            r3d = np.append(r3d, r3)

    return masses, x_arcsec, y_arcsec, r2d, r3d, np.array(redshifts)

def render_los_dynamic(rendering_class, aperture_radius, lens_plane_redshifts,
                       delta_zs, x_interp, y_interp, zmin, zmax):

    redshifts = []

    init = True

    for i, (zi, delta_zi) in enumerate(zip(lens_plane_redshifts, delta_zs)):

        if zi < zmin or zi > zmax:
            continue

        rescale_angle = rendering_class.rescale_angle(zi)

        # the rendering volume is automatically rescaled, pass in the baseline radius
        m = rendering_class.render_masses(zi, delta_zi, aperture_radius)
        x_shift, y_shift = x_interp(zi), y_interp(zi)

        x, y, r2, r3 = rendering_class.render_positions_at_z(zi, len(m), rescale_angle,
                                                             x_shift, y_shift)

        redshifts += [zi] * len(x)

        if init:
            masses = m
            x_arcsec, y_arcsec, r2d, r3d = x, y, r2, r3
            init = False
        else:
            x_arcsec = np.append(x_arcsec, x)
            y_arcsec = np.append(y_arcsec, y)
            masses = np.append(masses, m)
            r2d = np.append(r2d, r2)
            r3d = np.append(r3d, r3)

    return masses, x_arcsec, y_arcsec, r2d, r3d, np.array(redshifts)

def render_main(rendering_class):

    masses, x_arcsec, y_arcsec, r2d, r3d, redshifts = rendering_class()

    return masses, x_arcsec, y_arcsec, r2d, r3d, redshifts

def render_main_dynamic(rendering_class, aperture_size, center_x, center_y, mlow, mhigh):

    masses, x_arcsec, y_arcsec, r2d, r3d, redshifts = rendering_class(center_x, center_y, mlow, mhigh, aperture_size)

    return masses, x_arcsec, y_arcsec, r2d, r3d, redshifts

