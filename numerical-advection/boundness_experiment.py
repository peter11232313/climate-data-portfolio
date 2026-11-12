# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 15:08:14 2026

"""
import matplotlib.pyplot as plt
import numpy as np
import os
from TVD_scheme import TVD
from schemes_sws02hs import ARTD4
from advectionHelpers import savePlot, cosBell_0_05
from semiLagrangian import semiLagrangian

def test_boundness(schemes):
    """
    Function to test boundness for all schemes
    Parameters
    ----------
    schemes : List
        The scheme list to be tested
        
    Returns
    -------
    None. Instead plots a graph of how the max and min of phi at different 
    timestamp are bounded/not bounded and save the plot

    """
    c_test_list = [0.5, 1.0, 1.5]
    nt = 100
    nx = 40
    xLim = [0,1]
    dx = (xLim[1] - xLim[0])/nx
    x = np.arange(xLim[0], xLim[1], dx)
    phi_init = cosBell_0_05(x)
    phiLevels = [phi_init.copy()]
    phi_init_max = max(phi_init)
    phi_init_min = min(phi_init)
    
    
    for i in range(len(schemes)):
        for c in c_test_list:
            schemeResults = np.zeros((nt+1, nx))
            dt = c * dx
            retrieve_result_from_scheme(schemes[i], phiLevels.copy(), c, nt,\
                                        schemeResults)
            max_phi = np.max(schemeResults, axis=1)
            min_phi = np.min(schemeResults, axis=1)
            
            time_array = np.linspace(0.0, nt * dt, nt + 1)
        
            plt.figure(figsize=(8, 4.5))
            plt.plot(
                time_array,
                max_phi,
                label="Max $\\phi$",
                color="red",
                linewidth=1.5,
            )
            plt.plot(
                time_array,
                min_phi,
                label="Min $\\phi$",
                color="blue",
                linewidth=1.5,
            )
            plt.axhline(phi_init_max, color="black", linestyle=":", \
                        linewidth=0.8, label="max of $\\phi$ at time 0")
            plt.axhline(phi_init_min, color="black", linestyle="--",\
                        linewidth=0.8, label = "min of $\\phi$ at time 0")
            
            plt.title(f"scheme = {schemes[i].__name__}, max and min of phi : Courant Number c = {c}")
            plt.xlabel("Time ($t$)")
            plt.ylabel(r"Solution Bounds ($\phi$ values)")
            plt.legend(loc="upper left")
            plt.grid(True, linestyle=":", alpha=0.6)
            fig = plt.gcf()
            
            savePlot("boundness", fig, \
                     f"{schemes[i].__name__}_test_courant_{c}.png", \
                         schemes[i].__name__)
    return


 


def retrieve_result_from_scheme(scheme, phiLevels, c, nt, schemeResults):        
    """
    Function to store all results from different timestamp after running a scheme

    Parameters
    ----------
    scheme : function
        The scheme function.
    phiLevels ([1darray]): 
        A list length one containing a 1darray which is the latest value of phi,
        to be updated.
    c: float 
        The Courant number

    Returns
    -------
    None. Instead plots a graph of how the TV changes and prints the final
    TV of each scheme.

    """
    ##initial condition
    schemeResults[0] = phiLevels[0]
    j = 1
    for i in range(nt):
        scheme(phiLevels, c)
        schemeResults[j] = phiLevels[-1]
        j+=1
    return


# First make sure that the plots folder exists
if not os.path.exists("plots"):
    os.makedirs("plots")

# List of functions to do the advection
allSchemes = [ARTD4, semiLagrangian, TVD]
test_boundness(allSchemes)
