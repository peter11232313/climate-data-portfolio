# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 23:01:41 2026

Description: Code to perform TVD scheme
"""

import numpy as np


def vanLeer(r):
    '''
    Function to compute van Leer limiter
    Parameters:
        phi (phi: 1darray): 1d-array to store phi value at particular time
    Returns: 1darray
        values of Leer limiter (r + np.abs(r)) / (1.0 + np.abs(r))
    '''
    return (r + np.abs(r)) / (1.0 + np.abs(r))


def compute_phi_ratio(phi): 
    '''
    Function to compute ratio phi_i - phi_{i-1} / phi_{i+1} - phi_i
    For case of in difference <= small number eps, we treat as 0 and
    we have special handling. 
    In case denominator is < eps, there will be 2 cases
    1. nominator is also 0 ==> we assign 0 to the ratio
    2. nominator is not 0 ==> we take the sign of nominator and multiply by 1e6, which is a large number to demonstrate large raio
    Parameters:
        phi (phi: 1darray): 1d-array to store phi value at particular time
    Returns:
        r_local: 1darray
            ratio phi_i - phi_{i-1} / phi_{i+1} - phi_i at all locations

    '''
    eps = 1e-12
    upper_diff = phi - np.roll(phi, 1)          # phi_i - phi_{i-1}
    local_diff = np.roll(phi, -1) - phi         # phi_{i+1} - phi_i
    
    r_local = np.zeros_like(phi)
    local_diff_valid = np.abs(local_diff) > eps
    r_local[local_diff_valid] = upper_diff[local_diff_valid] / local_diff[local_diff_valid]
    
    local_diff_invalid = ~local_diff_valid
    upper_diff_zero = np.abs(upper_diff) <= eps
    r_local[local_diff_invalid & upper_diff_zero] = 0.0
    r_local[local_diff_invalid & ~upper_diff_zero] = np.sign(upper_diff[local_diff_invalid & ~upper_diff_zero]) * 1e6
    
    return r_local


def TVD(phiLevels, c):
    '''
    Parameters:
        phiLevels ([1darray]): A list length one containing a 1darray which is
                         the latest value of phi, to be updated.
        c (float): The Courant number
    Returns:
        None
    '''

    phi = phiLevels[0].copy()

    r_local = compute_phi_ratio(phi)
    r_upper = np.roll(r_local, 1)
    phi_prev = np.roll(phi, 1)    # phi[i-1]
    phi_next = np.roll(phi, -1)   # phi[i+1]

    vanLeer_local = vanLeer(r_local)
    vanLeer_upper = vanLeer(r_upper)

    phiNew = (phi
      - c * (phi - phi_prev)
      + 0.5 * c * (c - 1.0) * (vanLeer_local * (phi_next - phi)
                               - vanLeer_upper * (phi - phi_prev)))
    phiLevels.append(phiNew.copy())
    phiLevels.pop(0)
    return
