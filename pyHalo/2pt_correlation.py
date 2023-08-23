# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 14:01:57 2023

@author: Birendra Dhanasingham, The University of New Mexico, NM. 
"""
import numpy as np
from scipy.interpolate import RectBivariateSpline
import time
from scipy.integrate import simps
from scipy.special import eval_chebyt

"""
###############################################################################
Main classes and functions for xiHalo
###############################################################################
"""

def grids(numpix_x, numpix_y, window_size_x, window_size_y, r): 
    
    """
    :param numpix_x: Number of pixels in the window in the x-direction
    :param numpix_y: Number of pixels in the window in the y-direction
    :param window_size_x: Size of the window in the x-direction
    :param window_size_y: Size of the window in the y-direction
    :param r: 1D array of r-values
    :return : XX, YY: large grid, xx, yy: small grid
    
    """             
    rmin_max_x = window_size_x/2
    rmin_max_y = window_size_y/2
    x = np.linspace(-rmin_max_x, rmin_max_x, numpix_x)
    y = np.linspace(-rmin_max_y, rmin_max_y, numpix_y)
    xx, yy = np.meshgrid(x, y)

    Rmin_max_x = rmin_max_x + r.max() #Actually Rmin_max_x = rmin_max_x + r.max()/2
    Rmin_max_y = rmin_max_y + r.max() #Actually Rmin_max_y = rmin_max_y + r.max()/2
    Npix_x = np.int((numpix_x/rmin_max_x)*Rmin_max_x)
    Npix_y = np.int((numpix_y/rmin_max_y)*Rmin_max_y)
    X = np.linspace(-Rmin_max_x, Rmin_max_x, Npix_x)
    Y = np.linspace(-Rmin_max_y, Rmin_max_y, Npix_y)
    XX, YY = np.meshgrid(X, Y)

    return XX, YY, xx, yy


def mask_annular(center_x, center_y, x_grid, y_grid, r_min, r_max = None):
    """
    :param center_x: x-coordinate of center position of circular mask
    :param center_y: y-coordinate of center position of circular mask
    :param x_grid: x-coordinate grid
    :param y_grid: y-coordinate grid
    :param r_min: inner radius of mask in unit of grid coordinates
    :param r_max: outer radius of mask in unit of grid coordinates
    :return: mask array of shape x_grid with =0 inside the radius and =1 outside
    """
    x_shift = x_grid - center_x
    y_shift = y_grid - center_y
    R = np.sqrt(x_shift*x_shift + y_shift*y_shift)
    mask = np.ones(x_shift.shape)
    if r_max == None:
        mask[R <= r_min] = 0
    else:
        mask[R <= r_min] = 0
        mask[R >= r_max] = 0

    return mask


def corr_kappa_with_mask(kappa_map, XX_, YY_, xx_, yy_, r, mu, apply_mask = True, r_min = 0, r_max = None, normalization = True): 
    
    start_time = time.time()
    
    assert kappa_map.shape == XX_.shape, f"Convergence map must NOT be computed using  the window!"
    
    X_ = XX_[0]
    Y_ = YY_[:,0]

    x_ = xx_[0]
    y_ = yy_[:,0]
    
    phi = np.arctan2(yy_, xx_)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
        
    center_x = (X_.max()+ X_.min())/2
    center_y = (Y_.max()+ Y_.min())/2
    
    if apply_mask == True:
        mask = mask_annular(center_x, center_y, XX_, YY_, r_min, r_max)
        mask_interp = RectBivariateSpline(X_, Y_, mask, kx=1, ky=1, s=0)
        
    else:
        mask = np.ones(XX_.shape)
        
    
    kappa_interp = RectBivariateSpline(X_, Y_, kappa_map, kx=5, ky=5, s=0)
       
    corr = np.zeros((r.shape[0], mu.shape[0]))
       
    for i in range(r.shape[0]):
        for j in range(mu.shape[0]):
            x1 = xx_ - (r[i]/2)*((cos_phi*mu[j]) + (sin_phi*np.sqrt(1-mu[j]**2)))
            y1 = yy_ - (r[i]/2)*((sin_phi*mu[j]) - (cos_phi*np.sqrt(1-mu[j]**2)))
            
            x2 = xx_ + (r[i]/2)*((cos_phi*mu[j]) + (sin_phi*np.sqrt(1-mu[j]**2)))
            y2 = yy_ + (r[i]/2)*((sin_phi*mu[j]) - (cos_phi*np.sqrt(1-mu[j]**2)))
            
            if apply_mask == True:
                if r_max == None:
                    Area = (x_.max()-x_.min())*(y_.max()-y_.min()) - np.pi*(r_min**2) 
                else:
                    Area = np.pi*(r_max**2 - r_min**2)
            else:
                Area = (x_.max()-x_.min())*(y_.max()-y_.min())
            
            kappa_interp_1_ = kappa_interp(y1, x1, grid = False)
            if apply_mask == True:
                mask_interp_1 = mask_interp(y1, x1, grid = False)
            else:
                mask_interp_1 = np.ones(x1.shape)
            
            kappa_interp_2_ = kappa_interp(y2, x2, grid = False)
            if apply_mask == True:
                mask_interp_2 = mask_interp(y2, x2, grid = False)
            else:
                mask_interp_2 = np.ones(x2.shape)
                
            mask_interp_1[mask_interp_1<0.9] = 0
            mask_interp_1[mask_interp_1>0.9] = 1

            mask_interp_2[mask_interp_2<0.9] = 0
            mask_interp_2[mask_interp_2>0.9] = 1 
            
            kappa_interp_1 = kappa_interp_1_*mask_interp_1
            kappa_interp_2 = kappa_interp_2_*mask_interp_2
            
            term_1 = simps(simps(kappa_interp_1*kappa_interp_2, x_, axis=0), y_, axis=-1)
            term_2_1 = simps(simps(kappa_interp_1*mask_interp_2, x_, axis=0), y_, axis=-1)
            term_2_2 = simps(simps(mask_interp_1*kappa_interp_2, x_, axis=0), y_, axis=-1)
            term_3 = simps(simps(mask_interp_1*mask_interp_2, x_, axis=0), y_, axis=-1)
            
            NCC_num = (term_1 - ((term_2_1*term_2_2)/term_3))/Area
            
            term_4 = simps(simps((kappa_interp_1**2)*mask_interp_2, x_, axis=0), y_, axis=-1)
            term_5 = term_2_1**2
            term_6 = term_3
            
            NCC_den_1 = (term_4 - (term_5/term_6))/Area
            
            term_7 = simps(simps(mask_interp_1*(kappa_interp_2**2), x_, axis=0), y_, axis=-1)
            term_8 = term_2_2**2
            term_9 = term_3
            
            NCC_den_2 = (term_7 - (term_8/term_9))/Area   
            
            if normalization == True:
                corr[i,j] = NCC_num/np.sqrt(NCC_den_1*NCC_den_2) 
            else:
                corr[i,j] = NCC_num
            
    end_time = time.time()
    
    print(f"It took {end_time-start_time:.2f} seconds to compute the correlation map") 
    
    return corr


def xi_l(l, corr, r, mu):
    T_l = eval_chebyt(l, mu)  
    func = corr*T_l
    
    if l==0:
        prefactor = 1/np.pi
    else:
        prefactor = 2/np.pi
    
    xi_l = np.zeros(r.shape[0])
    for i in range(r.shape[0]):
        xi_l[i] = simps(func[i], np.flip(np.arccos(mu)), axis=0)
               
    return r, prefactor*xi_l 
