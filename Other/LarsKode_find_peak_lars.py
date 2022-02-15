from astropy.io import fits
from astropy.io.fits import getheader
from astropy.stats import sigma_clip
from scipy import optimize


import time    # only for cheking how long does it take to run a part of the script - especially the gaussian fit part

import numpy as np
from math import sqrt, log, e
import matplotlib.pyplot as plt

import csv

from astropy.modeling.fitting import LevMarLSQFitter, LinearLSQFitter, SimplexLSQFitter, SLSQPLSQFitter, FittingWithOutlierRemoval

from astropy.stats import SigmaClip

from scipy.signal import find_peaks, peak_widths, peak_prominences

from astropy.modeling.models import Gaussian1D, Chebyshev1D, Polynomial1D

import tkinter as tk
from tkinter import filedialog
from tkinter import *
import re
from repack import wlsol_from_header

import sys

def select_file():
    """selecting the file (or files - change to filedialog.askopenfilenames)"""
    root = tk.Tk()
    file_selection = filedialog.askopenfilename(parent=root, title='Choose file')
    root.destroy()
    # file = fits.getdata(file_selection)
    return file_selection


def func_find_peaks(y, dist):  # height, dist):
    """identifies peaks, input : y = an array that contains the signal - the flux/intensity etc; dist = minimum distance between peaks
     returns peak index, prominences, left and right indexes - for widths and the actual width
    [0] peak index in initial array, [1] prominences, [2] left index for widths, [3] right index for widths ,
    [4] length of the width, [5] peak height 0 reference """

    peaks, dict_peak = find_peaks(x=y, distance=dist, height=np.zeros(len(y)))  # height=height,
    prom = peak_prominences(x=y, peaks=peaks, wlen=20)
    widths = peak_widths(x=y, peaks=peaks, rel_height=1, prominence_data=prom)
    return peaks, prom[0], np.round(widths[2]).astype(int), np.round(widths[3]).astype(int), widths[0], dict_peak['peak_heights']


def gauss_fit_linear(x, y, a, c, std):  # ,weight):
    """  Gaussian + linear fitting
    x = pixel array that define the peak
    y = intensity array
    a = amplitude (guess - prominence from the peak find function)
     c = centre of the gaussian (guess - pixel corresponding to the prominence)
     std =std deviation of the gaussian (guess - 0.25 of the width of the peak) """
    gauss = Gaussian1D(amplitude=a, mean=c, stddev=std)
    p_init = Polynomial1D(degree=1)
    compound_model = gauss + p_init
    fitter = SimplexLSQFitter()
    gg_fit = fitter(compound_model, x, y, maxiter=500)  # weights=weight,
    gauss_centre = gg_fit[0].mean.value
    std = gg_fit[0].stddev.value
    return gauss_centre, gg_fit, std

def gauss_fit_linear_fast(pixelpos, flux, a, c, std):  # ,weight):
    """  Gaussian + linear fitting
    x = pixel array that define the peak
    y = intensity array
    a = amplitude (guess - prominence from the peak find function)
    c = centre of the gaussian (guess - pixel corresponding to the prominence)
    std =std deviation of the gaussian (guess - 0.25 of the width of the peak) """

    gaussfunc = lambda p, x: p[3] + p[4]*x + p[0] * np.exp(-((x - p[1]) / p[2])**2 / 2.0)
    errfunc = lambda p, x, y: gaussfunc(p, x) - y  # Distance to the target function
    p0 = [a, c, std, 0.0, 0.0]  # Initial guess for the parameters
    out = optimize.leastsq(errfunc, p0[:], args=(pixelpos, flux), full_output=1)
    pfinal = out[0]
    covar = out[1]

    return pfinal[1], np.sqrt( covar[1][1])

# just a gaussian
# def gauss_fit_linear(x, y, a, c, std):  # ,weight):
#     """  Gaussian + linear fitting
#     x = pixel array that define the peak
#     y = intensity array
#     a = amplitude (guess - prominence from the peak find function)
#      c = centre of the gaussian (guess - pixel corresponding to the prominence)
#      std =std deviation of the gaussian (guess - 0.25 of the width of the peak) """
#     gauss = Gaussian1D(amplitude=a, mean=c, stddev=std)
#     # p_init = Polynomial1D(degree=1)
#     # compound_model = gauss + p_init
#     fitter = SimplexLSQFitter()
#     gg_fit = fitter(gauss, x, y, maxiter=3000)  # weights=weight,
#     gauss_centre = gg_fit.mean.value
#     std = gg_fit.stddev.value
#     return gauss_centre, gg_fit, std


def cutting_peaks(lower_cut_off, upper_cut_off, index_peak_array,flux_of_peak, blaze_coeff):  #peaks_height
    ''' cuts out the peaks that are too low or too high #
        input: lower_cut_off = lower limit for a peak intensity - number -int or float - positive ,
               upper_cut_off = upper limiy for a peak intensity - number -int or float - positive,
               array_peaks_ind = array of indexes for peak result of func_find_peak,
               peaks_height = array with height of each peak,
               blaze_coeff = array that contains the blaze function values - from fits file - 3rd layer
        output: indexes of peaks that are inside the specified limits (lower and upper cut offs) '''
    lower = np.where(flux_of_peak[index_peak_array] * blaze_coeff[index_peak_array] > lower_cut_off)
    upper = np.where(flux_of_peak[index_peak_array] * blaze_coeff[index_peak_array] < upper_cut_off)
    return np.intersect1d(lower, upper)

# speed of light m/s
c = 299792458

# select LFC files or file

f = sys.argv[1]
# f = select_file()


# go through all selected files

#for f in selected_file: #(for more than one file)

# load the fits file
lfc = fits.getdata(f)

# load header - for the ThAr wavelength solution
header = getheader(f)

# function that will generate the ThAR wl solution
w = wlsol_from_header(header)

# wavelength from second fits layer
wlfc = lfc[1]

# intensity form first fits layer
intensity_undeblazed = lfc[0]

# blaze function from 3rd fits layer
blaze = lfc[2]

# removing the blaze function
intensity = intensity_undeblazed / blaze

# convert from wavelength to frequency - GHz  #
frequency_lfc = (c / (wlfc * 1e-10)) * 1e-9

# minimum distance between LFC peaks
distance_peak = 10

# pixel array #
pixels = np.linspace(start=0, stop=len(wlfc[1]) - 1, num=len(wlfc[1]))

# empty array for storing the peak centres
all_centres = np.array([[]])

# empty dictionary where the centres for each aperture are stored
echelle_dict = {}

# empty dictionary where the fwhm for peak from each aperture are stored
fwhm_dict = {}
from collections import OrderedDict

start = time.time()

# plt.figure()
# go through all echelle orders
for k in range(0, len(wlfc)):  # len(wlfc)

    # load intensity for the specified echelle order
    intensity_order = intensity[k]

    # find the peaks - store indexes, left and right pixel of the width, the width, the prominence and the height
    peaks_prom_width = func_find_peaks(intensity_order, distance_peak)

    # cutting off the peaks that are too low or too high #
    cut_offs = cutting_peaks(2000, 650000, peaks_prom_width[0], intensity_order, blaze[k])
    peaks_pixel = peaks_prom_width[0][cut_offs]
    left_width_pixel = peaks_prom_width[2][cut_offs]
    right_width_pixel = peaks_prom_width[3][cut_offs]
    widths = peaks_prom_width[4][cut_offs]
    prominences = peaks_prom_width[1][cut_offs]
    height_peaks = peaks_prom_width[5][cut_offs]

    # see which peaks were selected
    # plt.figure()
    # plt.plot(intensity_undeblazed[k])
    # plt.vlines(peaks_pixel, 0, intensity_undeblazed[k][peaks_pixel])


    # initialize empty lists for peak centres, gauss param, and selecred data # - only centre are necessary
    centre_pix = []
    # param_gauss = []
    data_x_all = []
    fwhm_all = []

    # plt.figure()

    # condition - more than 200 lfc peaks have to be found in an echelle order - otherwise it is
    # considered that the LFC is not in the frame
    if len(peaks_pixel) > 350:
        peak_diff =[]
        # go through all peaks
        for j in range(len(peaks_pixel)):
            # select data pixel and flux - for gaussian fitting
            data_pixel = np.linspace(start=left_width_pixel[j], stop=right_width_pixel[j],
                                     num=(right_width_pixel[j] - left_width_pixel[j]) + 1).astype(int)
            data_inten_n = intensity_order[data_pixel]
            data_inten = data_inten_n - np.min(data_inten_n)

            # condition for not fitting if the peak is defined by less than 9 pixels
            if len(data_pixel) <= 9 or len(data_inten_n) <= 9:
                # print('Too short dataset - wrong identification of peak')
                pass
            else:
                # fit the gaussian
                gauss_fit = gauss_fit_linear(data_pixel, data_inten, prominences[j], peaks_pixel[j],
                                                  widths[j] / 4)
                gauss_fit_fast = gauss_fit_linear_fast(data_pixel, data_inten, prominences[j], peaks_pixel[j],
                                             widths[j] / 4)

                # if the centre is negative or higher than 6600  don't store it
                if gauss_fit[0] < 0 or gauss_fit[0] > 6600:
                    pass
                else:
                    # print('fit difference peak number'+ str(j))
                    # print(np.trapz(y=data_inten, x=data_pixel)-np.trapz(y=gauss_fit[1](data_pixel), x=data_pixel))
                    # peak_diff.append(np.abs(np.trapz(y=data_inten, x=data_pixel)-np.trapz(y=gauss_fit[1](data_pixel), x=data_pixel))/(np.trapz(y=gauss_fit[1](data_pixel))))
                    centre_pix.append(gauss_fit[0])
                    # param_gauss.append(gauss_fit[1])
                    data_x_all.append(data_pixel)
                    fwhm_all.append(2*sqrt(-2*log(0.5, e))*gauss_fit[1])


                ######### plot lines - gaussians plotting #####


                 # #handles, labels = plt.gca().get_legend_handles_labels()
                 # #by_label = OrderedDict(zip(labels, handles))
                 # xx = np.linspace(start=left_width_pixel[j], stop=right_width_pixel[j], num=50)
                 # plt.plot(data_pixel, data_inten, 'b-', label ='Data')
                 # # plt.plot(centre_pix, gauss_fit[1](centre_pix),  'rX', label ='Peaks centres') #
                 # # plt.vlines(x=centre_pix, ymin = 0, ymax = gauss_fit[1](centre_pix), linestyles='dotted')
                 # plt.plot(xx, gauss_fit[1](xx), 'g-',label='Gaussian fits')
                 # plt.xlabel('Pixels', fontsize=26)
                 # plt.ylabel('Intensity [Counts]', fontsize=26)
                 # plt.tick_params(axis='both', which='major', labelsize=24)
                 # #plt.title('Échelle order ' + str(k+1))
                 # plt.legend(by_label.values(), by_label.keys(), fontsize='xx-large')

        # Load pixel centre and fwhm

        echelle_dict[k] = centre_pix
        fwhm_dict[k] = fwhm_all
        print(k)
    else:
        print('NO LFC lines for echelle order ' + str(k + 1))
# plt.title('number of iterations = 200')
end = time.time()
print(end - start)

# save fwhm in csv file

with open("fwhm_" +str(re.search(r'(.*)/(.*)', f).group(2)) +".csv", "w") as csvfwhm:
    w_fwhm = csv.writer(csvfwhm, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for key, val in fwhm_dict.items():
        w_fwhm.writerow([key, val])
# dictionary were the wavelength is loaded

wave_dict = {}
res_dict = {}
fit_res_dict ={}
# go through all echelle orders where there are peaks
for ech in range(len(echelle_dict)):

    # load aperture from dictionary key
    aperture = list(echelle_dict.keys())[ech]

    # load pixel centres in an array from dictionary values
    pixel_centre = np.array(echelle_dict[aperture])  # aperture

    # convert peak centres from pixel to wavelength with the function generated from header
    wave_pixel_centre = w(pixel_centre, aperture)


    #### plot lines - pixel as a function of wavelength - peak centres  ####
    # plt.figure()
    # plt.plot(pixel_centre, wave_pixel_centre, '.')
    # plt.xlabel('Pixel')
    # plt.ylabel('Wavelength[$\AA$]')
    # plt.title('échelle order ' + str(aperture + 1))

    # convert peak centres from wavelength to frequency
    frequency_pixel_centre = (c / (wave_pixel_centre * 1e-10)) * 1e-9

    # extract the peak numbers
    n_i_float = (frequency_pixel_centre - 6.19) / 14.0

    # rounding  up to integers
    n_i_int = np.round(n_i_float)

    # generate the theoretical frequencies
    freq_teo = (14 * n_i_int) + 6.19

    # compute the differences between the theoretical frequencies and the ones generated from the ThAr sol
    res_freq = freq_teo - frequency_pixel_centre

    # compute same difference but in wavelength
    res_wave = c * 10 / res_freq

    ## plot lines - plot pixel(x-axis) and wavelength (y-axis); ThAR vs LFC  ##
    # plt.figure()
    # plt.plot(pixels, wlfc[aperture], '.', label='Th Ar solution')
    # plt.plot(pixel_centre, wave_pixel_centre, '.', label='LFC Peaks')
    # plt.xlabel('Pixel')
    # plt.ylabel('Wavelength [$\AA$]')
    # plt.title('échelle order ' + str(aperture + 1))

    #   plot lines - 3 subplots - (1) - Difference between ThAr and LFC - frequency #
    #                             (2) - Th Ar frequencies  #
    #                             (3) - Theory Frequencies #

    # plt.figure()
    # ax = plt.subplot2grid((3, 1), (0, 0))
    # plt.plot(n_i_int, res_freq)
    # plt.ylabel('Residuals [$\Delta\AA$]')
    # # plt.ylabel('Delta frequency')
    # ax2 = plt.subplot2grid((3, 1), (1, 0))
    # plt.plot(n_i_int, frequency_pixel_centre, '.')
    # plt.ylabel('Frequency Th Ar (GHz)')
    # ax3 = plt.subplot2grid((3, 1), (2, 0))
    # plt.plot(n_i_int, freq_teo, '.')
    # plt.ylabel('Frequency theory (GHz)')
    # plt.xlabel('number of peak')
    # plt.title('échelle order ' + str(aperture + 1))

    # convert the theoretical frequencies from frequency to wavelength #
    wave_theo = c * 10 / freq_teo

    # fit polynomial for a wavelength solution  ##

    p_init = Chebyshev1D(degree=6)

    # linear fitter
    fitter = LinearLSQFitter()

    # sigma clip - function
    outlier_func = SigmaClip(sigma_lower=3.5, sigma_upper=3.5, maxiters=3)

    poly_fit = FittingWithOutlierRemoval(fitter, outlier_func, niter=3)

    # Chebyshev polynomial
    p_fit = poly_fit(p_init, pixel_centre, wave_theo)

    # Compute wavelength in integer pixels
    a = np.around(p_fit[0](pixels), decimals=12)

    # compute wavelength in peak centres
    a1 = p_fit[0](pixel_centre)

    fitting_residuals = np.around(a1 - wave_theo, decimals=12)

    fitting_residuals_list = list(fitting_residuals)

    fit_res_dict[aperture] = fitting_residuals_list



    #### plot lines - fitted, theory and th ar
    # plt.figure()
    # plt.plot(pixel_centre, a1, 'o', label = 'Fitted ')
    # plt.plot(pixel_centre, wave_theo, '.', label = 'theory')
    # plt.plot(pixel_centre, w(pixel_centre, aperture), '.', label = 'ThAr')
    # plt.title('echelle order '+ str(aperture))
    # plt.legend()



    # plot fitting residuals

    # fig, ax1 = plt.subplots(figsize=(20,12), dpi=300)
    #
    # ax1.set_xlabel('Wavelength [$\AA$]', fontsize=26)
    # ax1.set_ylabel('Fitting residuals[$\Delta\AA$]', fontsize=26)
    # ax1.plot(wave_theo, fitting_residuals, 'o')
    # plt.tick_params(axis='both', which='major', labelsize=24)
    #
    # # instantiate a second axes that shares the same x-axis - but with another unit -m/s
    #
    # ax2 = ax1.twinx()
    #
    # # residuals in m/s
    #
    # ms1 = c * fitting_residuals / wave_theo
    # ax2.set_ylabel('RV[m/s]', fontsize=26)
    # ax2.plot(wave_theo, ms1, 'o')
    # ax2.tick_params(axis='y', which='major', labelsize=24)
    # figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()
    #
    # fig.savefig('fitting_residuals_'+str(aperture+1)+'.png')
    # plot LFC peaks centres and integer peaks

    # plt.figure()
    # plt.plot(pixels, a, '-')
    # plt.plot(pixel_centre, wave_theo, '.')

    # difference between LFC wavelength solution and ThAr wavelength solution

    res = np.around(a - wlfc[aperture], decimals=12)

    res_list = list(res)
    res_dict[aperture] = res_list

    # plot lines # - difference between LFC wavelength solution and ThAr wavelength solution
    # left y axis - delta AA, right y axis - m/s, x axis - wavelength AA

    ###################################plot lines###################################################################
    # fig, ax1 = plt.subplots(figsize=(20,12), dpi=300)
    #
    # ax1.set_xlabel('Wavelength [$\AA$]', fontsize=26)
    # ax1.set_ylabel('Residuals[$\Delta\AA$]', fontsize=26)
    # ax1.plot(a, res, 'o')
    # plt.tick_params(axis='both', which='major', labelsize=24)
    #
    # # instantiate a second axes that shares the same x-axis - but with another unit -m/s
    #
    # ax2 = ax1.twinx()
    #
    # # residuals in m/s
    #
    # ms = c * res / a
    # ax2.set_ylabel('RV[m/s]', fontsize=26)
    # ax2.plot(a, ms, 'o')
    # ax2.tick_params(axis='y', which='major', labelsize=24)
    # figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()
    #
    # fig.savefig('thar_residuals_' + str(aperture + 1) + '.png')

    wave_dict[aperture] = a
    # save wavelength solution, pixel centres or theoretical wavelength of the pixel centres in txt files

    # np.savetxt("C:/Users/andig/Downloads/wavelength_solt"+str(list(echelle_dict.keys())[ech])+'_'+str(f)[25:]+'.txt', a, delimiter=",", fmt = '%10.8f')

    # np.savetxt("C:/Users/andig/Downloads/pixel_centre_"+str(list(echelle_dict.keys())[ech])+'_'+str(f)[25:]+'.txt', pixel_centre, delimiter=",", fmt = '%10.6f')

    # np.savetxt("C:/Users/andig/Downloads/wave_theo_"+str(list(echelle_dict.keys())[ech])+'_'+str(f)[25:]+'.txt', wave_theo, delimiter=",",  fmt = '%10.8f')
#



# export ThAr residuals in AA
with open("res_ThAR_1" +str(re.search(r'(.*)/(.*)', f).group(2)) +".csv", "w") as csvresthar:
    w_res = csv.writer(csvresthar, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for key, val in res_dict.items():
        w_res.writerow([key, val])

# fit residuals export in [AA]
with open("fit_res1" +str(re.search(r'(.*)/(.*)', f).group(2)) +".csv", "w") as csvresfit:
    w_fit = csv.writer(csvresfit, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for key, val in fit_res_dict.items():
        w_fit.writerow([key, val])



# write a layer in the Fits file with the LFC wavelength solution #
# write all layers in the cube + and add the fifth layer - then save it as fits file

# initialize an empty numpy array with 5 layers - 2nd (=88) and 3rd (=6600) dimensions are
# échelle order and number of pixels
cube = np.zeros((5,88,6600))

# load the same LFC file
hdu = fits.open(f)

# load the header of the LFC file

headermain = hdu[0].header

headermain.set('RP_LAYR4','lfcwave','Layer 4: LFC wavelength solution ' + '(ap ' + str(list(wave_dict.keys())[0]+1) + ' to '+ str(list(wave_dict.keys())[-1]+1) +')', after='RP_LAYR3')

# load the data
data = hdu[0].data[:,:,:]

# load the first 4 initial layers layers
cube[0,:,:] = data[0,:,:]
cube[1,:,:] = data[1,:,:]
cube[2,:,:] = data[2,:,:]
cube[3,:,:] = data[3,:,:]

# load the LFC wavelength solution in the last layer of the new fits file
cube[4,:,:] = data[1,:,:]

# load new wavelength solution just in the orders where the LFC peaks are found
for ee in list(wave_dict.keys()):
    cube[4,ee,:] = wave_dict[ee]


# create the fits file with 5 layers,
# file name inserts new in the old file name
# attention - the variable - f is a path string, so the re.search part removes everything after the last backslash

newfilepath = re.sub(r'(.spec)', r'.new', f)
newfilename = re.search(r'(.*)/(.*)', newfilepath).group(2)
fits.writeto(newfilename, data=cube, header=headermain)
