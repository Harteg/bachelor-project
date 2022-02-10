import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import stats
import math
from astropy.io import fits
import iminuit
from iminuit import Minuit
import sys
from astropy.utils.data import get_pkg_data_filename
from pylab import *
from sympy import false

sys.path.append('/Users/jakobharteg/Github/MyAppStat/')
from ExternalFunctions import Chi2Regression
from ExternalFunctions import nice_string_output, add_text_to_ax # useful functions to print fit results on figure

# from scipy.signal import find_peaks, peak_widths, peak_prominences
# originally from Lars
from scipy import signal
def func_find_peaks(y, required_dist, required_prominence):  # height, required_dist):
    """identifies peaks, input : 
        - y = an array that contains the signal - the flux/intensity etc; 
        - required_dist = minimum distance between peaks
        - required_prominence = minimum prominence of peaks
        
        returns peak index, prominences, left and right indexes - for widths and the actual width
        [0] peak index in initial array, [1] prominences, [2] left index for widths, [3] right index for widths ,
        [4] length of the width, [5] peak height 0 reference """

    peaks, dict_peak = signal.find_peaks(x=y, distance=required_dist, height=np.zeros(len(y)), prominence=required_prominence)  # height=height,
    prom = signal.peak_prominences(x=y, peaks=peaks, wlen=20)
    widths = signal.peak_widths(x=y, peaks=peaks, rel_height=1, prominence_data=prom)
    return peaks, prom[0], np.round(widths[2]).astype(int), np.round(widths[3]).astype(int), widths[0], dict_peak['peak_heights']


def get_peak_index_ranges(peak_locs, peak_range_size):
    """ 
    
    Returns
    -------

        peak_index_ranges : ndarray
            An array of indexes (start, end) of a range around each peak. 

     """
    peak_index_ranges = []
    for nPeak in peak_locs:
        start = nPeak - peak_range_size/2
        end = nPeak + peak_range_size/2
        peak_index_ranges.append([int(start), int(end)])
    return np.asarray(peak_index_ranges)


def fit_peaks(data_spec, data_spec_err, peak_index_ranges, print=False):
    """ Returns array of the fit values: A, mu, sigma, C, chi2_val """

    peak_fits = []
    for peak_index_range in peak_index_ranges:

        index_start, index_end = peak_index_range

        x = np.arange(index_start, index_end) # range from the start of the peak index to the end of the peak index (index ~ nPix)
        y = np.array(data_spec[index_start:index_end])
        ey = np.array(data_spec_err[index_start:index_end])

        # Fitting functions:
        def func_GaussConst(x, A, mu, sigma, C) :
            return A * np.exp(-0.5 * (((x)-mu)/sigma)**2)  +  C


        # ChiSquare fit model:
        def model_chi2(A, mu, sigma, C) :
            y_fit = func_GaussConst(x, A, mu, sigma, C)
            chi2 = np.sum(((y - y_fit) / ey)**2)
            return chi2
        model_chi2.errordef = 1

        # Fit peak with a Gaussian:
        A_init     = 0.45
        mu_init    = np.mean(x)
        sigma_init = -1.6
        C_init     = 0.02
        minuit = Minuit(model_chi2, A=A_init, mu=mu_init, sigma=sigma_init, C=C_init)

        # Perform the actual fit (and save the parameters):
        m = minuit.migrad()                                             
        # print(m)
        
        # Extract the fitting parameters and their uncertainties:
        mu_fit = minuit.values['mu']
        sigma_mu_fit = minuit.errors['mu']
        Npoints = len(x)
        Chi2_val = minuit.fval # The chi2 value
        
        # peak_fits.append([*minuit.values, Chi2_val])
        peak_fits.append(
            [minuit.values['A'],
            minuit.values['mu'],
            minuit.values['sigma'],
            minuit.values['C'],
            Chi2_val]
        )
        
        if print:
            print(f"  Peak fitted. N = {Npoints:2d}   Chi2 ={Chi2_val:5.1f}   Wave mean = {mu_fit:8.3f}+-{sigma_mu_fit:5.3f}")

    return np.asarray(peak_fits)


def get_true_wavel(data_wavel_given, peak_locs):
    """ Takes in the given wavelengths for each pixel and the list of peaks in pixel space.
        
        Returns
        -------

            List of the true wavelengths

     """

    # For a given wavel_given find the closest wavel_true solution 
    # Start by generating at least enough values of true_wavel
        
    def true_wavel(n):
        # True frequency
        c = 299792458   # m/s
        v_rep, v_offset = 14e9, 6.19e9
        f = v_rep * n + v_offset

        # True wavelength in ångstrøm
        wavel = c/f * 1e10
        return wavel

    wavel_given = data_wavel_given[peak_locs]
    wavel_given_low = wavel_given[0]
    wavel_given_high = wavel_given[-1]

    # gen true wavel
    wavel_true = [true_wavel(0), true_wavel(1)]
    n = 1
    while wavel_true[-1] > wavel_given_low * (1 - 0.001): # continue until we get well below the lowest given wavel. 
        n += 1
        wavel_true.append(true_wavel(n))


    # Cut off well above wavel_given_high
    wavel_true = np.asarray(wavel_true)
    wavel_true = wavel_true[wavel_true < wavel_given_high * (1 + 0.001)]
    wavel_true = wavel_true[::-1] # reverse order

    # wavel_true list is now quite a bit longer than peak_locs, but contains all of them
    # Now, find the closest match in wavel_true for each value in 
    wavel_true_match = []
    for lambd_given in wavel_given:
        wavel_true_match.append(min(wavel_true, key=lambda x:abs(x-lambd_given)))

    return wavel_true_match


# TODO :: rename this 
def peak_position_fit_func(x, c0, c1, c2, c3, c4, c5, c6):
    # def fit_func(x, c0, c1, c2, c3, c4, c5):
    # def fit_func(x, c0, c1, c2, c3, c4):
    # def fit_func(x, c0, c1, c2, c3):
    # def fit_func(x, c0, c1, c2):
    # def fit_func(x, c0, c1):
        return c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4 + c5*x**5 + c6*x**6
        return c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4 + c5*x**5
        return c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4
        return c0 + c1*x + c2*x**2 + c3*x**3
        return c0 + c1*x + c2*x**2
        return c0 + c1*x


# TODO :: rename this
def fit_peak_positions(wavel_true_match, peak_fits):
    x = np.asarray(peak_fits)[:,1]
    y = wavel_true_match
    ey = np.sqrt(wavel_true_match) * 0.001  # photon noise is possion (?) 

    # Plot data in errorbars
    # figPeak, axPeak = plt.subplots(figsize=(16, 8))
    # # axPeak.errorbar(x, y, ey, fmt='none', ecolor='k', linewidth=1, capsize=2, capthick=1)
    # # axPeak.errorbar(x, y, fmt='.', ecolor='k', linewidth=1, capsize=2, capthick=1, label="Data")
    # axPeak.plot(x, y, ".", label="Data")

    # Quad fit
    model_chi2 = Chi2Regression(peak_position_fit_func, x, y, ey)
    model_chi2.errordef = 1

    # Fit peak with a Gaussian:
    minuit = Minuit(model_chi2, c0=1, c1=1, c2=1, c3=1, c4=1, c5=1, c6=1)

    # Perform the actual fit (and save the parameters):
    minuit.migrad()                                             
        
    # Extract the fitting parameters and their uncertainties:
    # Npoints = len(x)
    # Nvar = 2                                        # Number of variables
    # Ndof_fit = Npoints - Nvar                       # Number of degrees of freedom = Number of data points - Number of variables
    # Chi2_fit = minuit.fval                          # The chi2 value
    # Prob_fit = stats.chi2.sf(Chi2_fit, Ndof_fit)    # The chi2 probability given N degrees of freedom
    # print(f"  Peak fitted. N = {Npoints:2d}   Chi2 ={Chi2_fit:5.1f}")

    return minuit.values

