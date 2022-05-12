from curses import A_INVIS
from tkinter import X
import numpy as np
from scipy import stats
from astropy.io import fits
from iminuit import Minuit
from pylab import *
from scipy.interpolate import interp1d
from os import walk
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from sympy import false
from calibration_functions import *
from tqdm import tqdm
import pandas as pd

sys.path.append('/Users/jakobharteg/Github/MyAppStat/')
from ExternalFunctions import nice_string_output, add_text_to_ax # useful functions to print fit results on figure

def angstrom_to_velocity(wavelength_shift):
    """ Converts wavelenth shift in angstrom to velocity shift in cm/s """
    c = 29979245800
    angstrom_to_cm = 1e-8
    return wavelength_shift * angstrom_to_cm * c


def velocity_to_angstrom(velocity):
    """ Converts velocity in cm/s to wavelength in angstrom"""
    c = 29979245800
    angstrom_to_cm = 1e-8
    return velocity / angstrom_to_cm / c


def make_nan_matrix(size):
    matrix = np.empty((size,size))
    matrix[:] = np.nan
    return matrix



SPECTRA_PATH_34411 = "/Users/jakobharteg/Data/34411_spectra/"
SPECTRA_PATH_10700 = "/Users/jakobharteg/Data/10700_spectra/"
SPECTRA_PATH_26965 = "/Users/jakobharteg/Data/26965_spectra/"
SPECTRA_PATH_101501 = "/Users/jakobharteg/Data/101501_spectra/"
SPECTRA_PATH_51PEG = "/Users/jakobharteg/Data/51peg/"

def get_all_spectra_filenames(spectra_path = SPECTRA_PATH_34411):
    """ Returns all filenames of the spectra files in the path speficied in spectra_path"""
    SPEKTRA_filenames = next(walk(spectra_path), (None, None, []))[2]  # [] if no file
    SPEKTRA_filenames = sorted(SPEKTRA_filenames)
    SPEKTRA_filenames = [spectra_path + x for x in SPEKTRA_filenames] # add path to the filename
    return SPEKTRA_filenames

def load_spectra_fits(filename):
    """ Returns data from a fits file with a given name """
    hdul = fits.open(filename)
    data = hdul[1].data.copy()
    hdul.close()
    return data

def get_spec_wavel(data, order, continuum_normalized=False, angstrom=False):
    """ Returns intensity, intensity_err and wavelength for a given spectra data.
        Use in conjunction with load_spectra_fits """

    excalibur_mask  = data['EXCALIBUR_MASK'][order]    # filter by EXCALIBUR_MASK
    data_spec       = data['spectrum'][order][excalibur_mask]
    data_spec_err   = data['uncertainty'][order][excalibur_mask]
    data_wavel      = data['BARY_EXCALIBUR'][order][excalibur_mask]

    if angstrom == False:
        data_wavel = angstrom_to_velocity(data_wavel) # convert angstrom to cm/s

    if continuum_normalized:
        cont = data['continuum'][order][excalibur_mask]
        data_spec = data_spec / cont
        data_spec_err = data_spec_err / cont

    return data_spec, data_spec_err, data_wavel


def get_spektra_date(filename):
    """ Returns the date of observation for a given fits filename """
    hdul = fits.open(filename)
    header = hdul[0].header
    hdul.close()
    date = header["DATE-OBS"]
    date = date[:date.index(" ")]
    return date


def get_spectra_dates(filenames):
    """ Returns a list of dates (year, month, date) for given list of fits filenames """
    dates = []
    for i in np.arange(len(filenames)):
        date = get_spektra_date(filenames[i])
        year, month, date = date.split("-")
        year, month, date = int(year), int(month), int(date)
        dates.append((year, month, date))
    return dates


def convert_dates_to_relative_days(dates):
    from datetime import datetime
    """ Returns a list of days relative to the first day """
    days = []
    first_day = datetime(*dates[0])
    for i in np.arange(len(dates)):
        d1 = datetime(*dates[i])
        dd1 = d1 - first_day
        days.append(dd1.days)
    return np.asarray(days)


# There are often several observations per night
def get_spectra_filenames_without_duplicate_dates(spectra_path = SPECTRA_PATH_34411):
    """ Returns a list of filesnames of the spectra files in the path speficied in SPECTRA_PATH_34411 without date-duplicates, i.e.
        oberservations taken on the same day. """
    all_files = get_all_spectra_filenames(spectra_path)
    all_dates = get_spectra_dates(all_files)
    files = [all_files[0]]
    dates = [all_dates[0]]
    for i in np.arange(1, len(all_dates)):
        if dates[-1] != all_dates[i]:
            dates.append(all_dates[i])
            files.append(all_files[i])
    return files


def plot_spectra_dates(spectra_dates, label=None):
    from datetime import datetime
    plt.figure(figsize=(25, 8))
    for index, date in enumerate(spectra_dates):
        year, month, date = date
        d = datetime(year, month, date)
        if label and index == len(spectra_dates) - 1:
            plt.scatter(d, index, color="k", s=2, label=label)
            plt.legend()
        else:
            plt.scatter(d, index, color="k", s=2)
        

def plot_spectra_dates_from_path(path):
    filenames = get_all_spectra_filenames(path)
    dates = get_spectra_dates(filenames)
    plot_spectra_dates(dates, label=f"{len(dates)} files found in {path}")



def plot_matrix(diff_matrix, diff_matrix_err=None, diff_matrix_valid=None, unit_cms=False, plot_ratio=True):
    """ Plot shift matrices """

    if plot_ratio:
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(8,3))
    else:
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,4))

    def fix_grid_lines(ax, size):
        ax.hlines(y=np.arange(0, size)+0.5, xmin=np.full(size, 0)-0.5, xmax=np.full(size, size)-0.5, color="black", alpha=0.2)
        ax.vlines(x=np.arange(0, size)+0.5, ymin=np.full(size, 0)-0.5, ymax=np.full(size, size)-0.5, color="black", alpha=0.2)


    # ======= PLOT 1 ============ Mean shifts
    cs = ax1.imshow(-diff_matrix)
    cax = make_axes_locatable(ax1).append_axes('right', size='5%', pad=0.05) # to make color bar same height at plot, needed when making several subplots with each colorbar
    cbar = fig.colorbar(cs, ax=ax1, cax=cax)
    if unit_cms:
        cbar.set_label('cm/s', rotation=270)
    else:
        cbar.set_label('angstrom', rotation=270)
    fix_grid_lines(ax1, len(diff_matrix))
    ax1.set_title("Wavelength shift")

    # ======= PLOT 2 ============ Errors
    if diff_matrix_err is not None:
        cs = ax2.imshow(diff_matrix_err)
        cax = make_axes_locatable(ax2).append_axes('right', size='5%', pad=0.05) # to make color bar same height at plot, needed when making several subplots with each colorbar
        cbar = fig.colorbar(cs, ax=ax2, cax=cax)
        if unit_cms:
            cbar.set_label('cm/s', rotation=270)
        else:
            cbar.set_label('angstrom', rotation=270, labelpad=15)
        fix_grid_lines(ax2, len(diff_matrix_err))
        ax2.set_title("Error")

    # ======= PLOT 3 ============ Valid/Convergence ratio
    if diff_matrix_valid is not None:
        if plot_ratio:
            cs = ax3.imshow(diff_matrix_valid)
            cax = make_axes_locatable(ax3).append_axes('right', size='5%', pad=0.05) # to make color bar same height at plot, needed when making several subplots with each colorbar
            cbar = fig.colorbar(cs, ax=ax3, cax=cax)
            fix_grid_lines(ax3, len(diff_matrix_valid))
            ax3.set_title("Valid ratio")
            # Fix spacing between plots
            fig.subplots_adjust(wspace=0.25)


    # fig.subplots_adjust(wspace=0)
    fig.tight_layout()
    # fig.savefig("matrx.pdf", bbox_inches="tight", dpi=300)




# def find_features(filename, plot_orders = None, plot_features_in_order = None, log=True):
def find_features(filename, 
                    plot_orders = None, 
                    plot_features_in_order = None, 
                    log=True, 
                    max_frac_err = 0.1,                 # maximum fractional error in intensity
                    min_order_goodness = 0.7,           # Min fraction of data in an order that should be left after filtering for the order to be included. 
                    min_peak_dist = 50,                 # minimum distance (in pixels) between peaks  
                    min_peak_prominence = 0.25,         # minimum height of peak from base (not zero)
                    is_51_peg = False,                  # 51Peg doesn't have excalibur column so we need different keywords 
                    velocity_cms = False,               # if true, will return wavelengths in cm/s
                    dont_use_bary_centric = False       # True if you want to use "wavelength" column filtered by pixel_mask.
    ):
    
    """ Returns list of features x_values (angstrom by default), y_values, y_err_values, x_peak_location, peak_index, order """

    
    feature_slices = []
    fits_data = load_spectra_fits(filename)
    orders_n = shape(fits_data["spectrum"])[0]

    # 51Peg data files have different column names
    is_51_peg = ("peg" in filename)

    orders = np.arange(0, orders_n)
    for order in orders:
        
        if is_51_peg:
            pixel_mask  = fits_data['PIXEL_MASK'][order]
            y           = fits_data['spectrum'][order][pixel_mask]
            og_y        = y # copy of original y data before filtering
            y_err       = fits_data['uncertainty'][order][pixel_mask]
            continuum   = fits_data['continuum'][order][pixel_mask]
            x           = fits_data['bary_wavelength'][order][pixel_mask]
        elif dont_use_bary_centric:
            pixel_mask  = fits_data['PIXEL_MASK'][order]    # filter by pixel_mask
            y           = fits_data['spectrum'][order][pixel_mask]
            og_y        = y # copy of original y data before filtering
            y_err       = fits_data['uncertainty'][order][pixel_mask]
            continuum   = fits_data['continuum'][order][pixel_mask]
            x           = fits_data['wavelength'][order][pixel_mask]
        else:
            excalibur_mask  = fits_data['EXCALIBUR_MASK'][order]    # filter by EXCALIBUR_MASK
            y           = fits_data['spectrum'][order][excalibur_mask]
            og_y        = y # copy of original y data before filtering
            y_err       = fits_data['uncertainty'][order][excalibur_mask]
            continuum   = fits_data['continuum'][order][excalibur_mask]
            x           = fits_data['BARY_EXCALIBUR'][order][excalibur_mask]

        # skip order if no good data
        if len(x) == 0 or len(y) == 0:
            continue

        # Normalize intensity by continuum 
        y = y/continuum
        y_err = y_err/continuum

        # Convert angstorm to cm/s
        if velocity_cms:
            x = angstrom_to_velocity(x) # Don't convert just yet

        # filter by fractional error 
        frac_err = y_err/y
        frac_err_mask = (0 < frac_err) & (frac_err < max_frac_err) # reject if larger than 10% and negative
        y = y[frac_err_mask]
        y_err = y_err[frac_err_mask]
        x = x[frac_err_mask]

        # Skip order if we filtered out more than 1 - min_order_goodness of the data (bad order ... )
        if len(y) / len(og_y) < min_order_goodness:
            # print("Order sucks:", len(y), "/", len(og_y), "=", len(y) / len(og_y))
            continue

        # Now invert peaks
        y = 1 - y

        peaks = func_find_peaks(y, min_peak_dist, min_peak_prominence)
        peak_locs = peaks[0]
        peak_height = peaks[5] # peak height from y=0 
        
        # Plot
        if plot_orders is not None and (plot_orders == order).any() or plot_orders == -1:
            plt.figure(figsize=(30,3))
            plt.plot(x, y, ".")
            plt.plot(x[peak_locs], peak_height, "o", color="C3", label=f"{order}. order")
            plt.ylabel("1 - Continuum normalized counts")
            if velocity_cms:
                plt.xlabel("Wavelength [cm/s]")
            else:
                plt.xlabel("Wavelength [Å]")
            plt.legend(loc = "upper right")
            
            # plt.figure(figsize=(30,1))
            # plt.plot(x_gone, frac_err[frac_err_mask == False], color="red")
            # plt.xlabel("Wavelength [Å]")

        # If less than 10 peaks skip order
        if len(peak_locs) < 5:
            continue

        peak_index_ranges = get_peak_index_ranges(peak_locs, peak_range_size=50)

        # feature_slices = []
        for index, range in enumerate(peak_index_ranges):
            start, end = range

            if len(x[start:end]) != 0 and len(y[start:end]) != 0: # sometimes these lists come out empty. 
                feature_slices.append([x[start:end], y[start:end], y_err[start:end], x[peak_locs[index]], peak_locs[index], order])

            if plot_features_in_order is not None and (plot_features_in_order == order).any():
                plt.figure(figsize=(10,3))
                plt.plot(x[start:end], y[start:end], ".")
                plt.plot(x[peak_locs[index]], peak_height[index], "o", color="C3", label=f"$k = {index}$")
                plt.legend(loc = "upper right")
        
    if log:
        print(len(feature_slices), "peaks found")

    return np.asarray(feature_slices, dtype=object)



def find_feature_matches(features1, features2, log=True, filter=True, relative_tolerance=0.0008):
    """ Finds features in two files, and iterates from the lowest wavelength to the highest, in small
        steps, asking if either of the files have features close to the current wavelength within a
        defined tolerance. If so, it returns a match. 
    """

    peaks1 = np.array(features1[:, 3], dtype=float)
    peaks2 = np.array(features2[:, 3], dtype=float)

    max1, max2 = np.nanmax(peaks1), np.nanmax(peaks2)
    min1, min2 = np.nanmin(peaks1), np.nanmin(peaks2)

    common_max = min([max1, max2])
    common_min = max([min1, min2])

    # We should of course only use each peak once, so let's keep track of the peaks we use
    used_peaks1 = np.asarray([False] * len(peaks1))
    used_peaks2 = np.asarray([False] * len(peaks2))

    range = np.linspace(common_min, common_max, 10000)
    # relative_tolerance = 0.0001 # lower value --> fewer matches, higher --> worse matches # but first we set a high value and then filter later
    # relative_tolerance = 0.00005 # lower value --> fewer matches, higher --> worse matches
    # relative_tolerance = 0.001 # lower value --> fewer matches, higher --> worse matches
    # relative_tolerance = 0.00002 # lower value --> fewer matches, higher --> worse matches
    
    matches = []
    for wavel in range:
        
        # see if we have a match in the two lists:
        match_mask1 = np.isclose(peaks1, wavel, relative_tolerance)
        match_mask2 = np.isclose(peaks2, wavel, relative_tolerance)
        
        # Only take match if we have only one match from each file
        if len(match_mask1[match_mask1]) == 1 and len(match_mask2[match_mask2]) == 1:

            # get the two features that matched
            f1 = features1[match_mask1][0]
            f2 = features2[match_mask2][0]

            # Now perform a bit of filtering: 
            if filter:

                # Check if the sum of the y values are about equal, if so it means we probably have the same peak.
                f1_spec, f2_spec = f1[1], f2[1]
                if np.isclose(np.sum(f1_spec), np.sum(f2_spec), 0.1) == False:
                    continue

                # If the difference bewteen the two peaks is larger than 5 cm/s pass it
                # f1_peak, f2_peak = f1[3], f2[3]
                # if abs(f1_peak - f2_peak) > 5:
                #     continue
                    
            # Get the index of the peak
            used_peak_index2 = np.where(match_mask2)[0]
            used_peak_index1 = np.where(match_mask1)[0]

            # Append if we have not already used this peak
            if used_peaks1[used_peak_index1] == False and used_peaks2[used_peak_index2] == False:
                
                # Append
                matches.append([f1, f2])

                # Mark that we used the peaks
                used_peaks1[used_peak_index1] = True
                used_peaks2[used_peak_index2] = True
            
    if log:
        print(f"{len(matches)} matches found")
    
    return matches


def nth_cloest_match(value, array, n):
    """ Returns the index of the nth closest match for a given value and a given array"""
    diff_array = np.abs(array - value)
    return np.argpartition(diff_array, n)[n]


def find_feature_matches2(features1, features2, apply_iqr_filter=False, return_iqr_filter=False):
    """ This feature match finder simply returns the closest match between two lists of features.
        
        set apply_iqr_filter to discard matches where the peak location difference is outside the inter 
        quartile range. 
        
        set return_iqr_filter to return a mask with TRUE for matches with a peak location difference inside
        the inter quartile range. 

        """

    peaks1 = np.array(features1[:, 3], dtype=float)
    peaks2 = np.array(features2[:, 3], dtype=float)

    # We should of course only use each peak once, so let's keep track of the peaks we use
    used_peaks1 = np.asarray([False] * len(peaks1))
    used_peaks2 = np.asarray([False] * len(peaks2))

    # find matches
    matches = []
    for i in np.arange(len(peaks1)):
        peak1 = peaks1[i]
        min_index = nth_cloest_match(peak1, peaks2, 0)

        # Append if we have not already used this peak
        if used_peaks1[i] == False and used_peaks2[min_index] == False:
            
            # Append
            matches.append([features1[i], features2[min_index]])

            # Mark that we used the peaks
            used_peaks1[i] = True
            used_peaks2[min_index] = True
    matches = np.asarray(matches)

    # Filter
    if apply_iqr_filter or return_iqr_filter:
        peak_loc_diffs = np.asarray([f1[3] - f2[3] for f1, f2 in matches])
        iqr_min, iqr_max = compute_IQR_bounds(peak_loc_diffs)
        iqr_mask = (peak_loc_diffs > iqr_min) & (peak_loc_diffs < iqr_max)

    if apply_iqr_filter:
        matches = matches[iqr_mask]

    if return_iqr_filter:
        return matches, iqr_mask

    return matches


def plot_feature_matches2(matches, mask):
    """ Plots an overview of the peak location difference from features matches with a valid/invalid mask """
    plt.figure(figsize=(10, 16))
    peak_loc_diffs = np.asarray([f1[3] - f2[3] for f1, f2 in matches])
    y = np.arange(len(peak_loc_diffs))
    plt.plot(peak_loc_diffs[mask], y[mask], ".", color="C2", label=f"{len(peak_loc_diffs[mask])} valid matches")
    plt.plot(peak_loc_diffs[mask == False], y[mask ==  False], ".", color="Red", label=f"{len(peak_loc_diffs[mask==False])} invalid matches")
    plt.xlabel("Peak shift [A]")
    plt.ylabel("Peak index")
    plt.gca().invert_yaxis()
    plt.legend(bbox_to_anchor=(1.45, 0.99))
    plt.title("Initial peak match selection with IQR filter")


def compute_feature_shift(x1, y1, y1_err, peak1, x2, y2, peak2, plot=False, ax=None):
    """ Attempts to fit two features with based on a shift parameter.
        Returns shift_min_final, shift_min_final_err, valid 
        """

    # Add shift for self test 
    # x2 = x2 + 0.25
    
    # Interp first file
    f1 = interp1d(x1, y1, kind='cubic', fill_value="extrapolate")
    f1_upper_err = interp1d(x1, y1 + y1_err, kind='cubic', fill_value="extrapolate")
    f1_lower_err = interp1d(x1, y1 - y1_err, kind='cubic', fill_value="extrapolate")

    c = 299792458 # m/s

    # ChiSquare fit model:
    def model_chi2(A):

        # Interpolate template
        # interp_x2 = x2 + A
        interp_x2 = x2 * (1 + A/c) # this should give proper RV, the wavelength should be stretched by a factor of (1 + v/c)
        f2 = interp1d(interp_x2, y2, kind='cubic', fill_value="extrapolate")

        # Find common x-range
        xmin = max([min(x1), min(interp_x2)])
        xmax = min([max(x1), max(interp_x2)])
        xnewCommon = np.linspace(xmin, xmax, 1000)
        
        # Evaluate interpolation
        ynew1 = f1(xnewCommon)
        ynew2 = f2(xnewCommon)

        # Evalute error interpolation
        ynew1_upper_err = f1_upper_err(xnewCommon)
        ynew1_lower_err = f1_lower_err(xnewCommon)

        ynew1_upper_err_abs = np.abs(ynew1 - ynew1_upper_err)
        ynew1_lower_err_abs = np.abs(ynew1 - ynew1_lower_err)
        ynew1_err = [np.mean([a,b]) for a, b in zip(ynew1_upper_err_abs, ynew1_lower_err_abs)]
        
        # Compute chi2
        chi2 = np.sum(((ynew1 - ynew2) / ynew1_err)**2)
        return chi2
    model_chi2.errordef = 1
        
    A_init = (peak2 / peak1 - 1 ) * c # shift between the two peaks

    # Compute bounds on A
    x1_min, x1_max = min(x1), max(x1)
    A_lower_bound = (x1_min / peak2 - 1 ) * c
    A_upper_bound = (x1_max / peak2 - 1 ) * c

    minuit = Minuit(model_chi2, A=A_init)
    minuit.limits["A"] = (A_lower_bound, A_upper_bound)
    minuit.migrad()
    
    # Results
    valid = minuit.valid
    shift_min_final = minuit.values['A']
    shift_min_final_err = minuit.errors['A']
    forced = minuit.fmin.has_made_posdef_covar
    at_limit = minuit.fmin.has_parameters_at_limit

    if forced or at_limit:
        valid = False

    
    # Plot final shifted values
    if plot:

        if ax == None:
            fig, ax = plt.subplots(figsize=(14,6))

        ax.plot(x1, y1)
        ax.plot(x2 * (1 + shift_min_final/c), y2)

        if not valid:
            ax.set_facecolor('pink')


    return shift_min_final, shift_min_final_err, valid #, minuit


def compute_all_feature_shifts(matches, log=True, plot=False, ax=[]):
    """ Calls compute_feature_shift for a list of matches """

    shifts = []
    # for k in tqdm(np.arange(len(matches))):
    for k in np.arange(len(matches)):
        f1 = matches[k][0]
        f2 = matches[k][1]

        x1      = f1[0]
        y1      = f1[1]
        y1_err  = f1[2]
        peak1   = f1[3]
        x2      = f2[0]
        y2      = f2[1]
        peak2   = f2[3]

        if len(ax) == 0:
            shifts.append(compute_feature_shift(x1, y1, y1_err, peak1, x2, y2, peak2, plot=plot))
        else:
            shifts.append(compute_feature_shift(x1, y1, y1_err, peak1, x2, y2, peak2, plot=plot, ax=ax[k]))


    shifts = np.asarray(shifts, dtype=object)
    if log:
        print(f"{len(shifts[:, 2][shifts[:, 2] == 0])} / {len(shifts)} fits failed")
    return shifts


def analyse_and_plot_shifts(path, file_index1, file_index2, bary=True):
    """ Short hand function to compute and plot shifts between two files """

    filenames = get_spectra_filenames_without_duplicate_dates(path)
    file1, file2 = filenames[file_index1], filenames[file_index2]
    matches = find_feature_matches(find_features(file1, dont_use_bary_centric=bary), find_features(file2, dont_use_bary_centric=bary))

    # Compute how many rows we need to display all
    height = 1
    while 10 * height < len(matches):
        height += 1
    assert (height < 100), "Height is more than 100"

    # Compute a plot shifts
    fig, axs = plt.subplots(height, 10, figsize=(20, 20))
    shifts = compute_all_feature_shifts(matches, plot=True, ax=axs.flat)

    # Remove ticks
    for ax in axs.flat:
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])

    # Plot shift values
    plot_features_shift(shifts)

    return shifts


def plot_features_shift(shifts, ax=None, labels=True):

    s, s_err, s_valid = shifts[:, 0], shifts[:, 1], shifts[:, 2]

    feature_n = np.arange(len(s))
    valid_shifts, valid_shifts_err, valid_features          = s[s_valid == 1], s_err[s_valid == 1], feature_n[s_valid == 1]
    invalid_shifts, invalid_shifts_err, invalid_features    = s[s_valid == 0], s_err[s_valid == 0], feature_n[s_valid == 0]

    if ax == None:
            fig, ax = plt.subplots(figsize=(8, 12))

    ax.errorbar(valid_shifts, valid_features, xerr=valid_shifts_err, fmt="none", linewidth=1, color = "C2", label=rf'{len(valid_shifts)} valid fits, with errors')
    ax.errorbar(invalid_shifts, invalid_features, xerr=invalid_shifts_err, fmt="none", linewidth=1, color = "C3", label=rf'{len(invalid_shifts)} invalid fits, with errors')

    ax.scatter(valid_shifts, valid_features, s=1)
    ax.scatter(invalid_shifts, invalid_features, s=1)


    shift_mean, shift_mean_err = weighted_mean(valid_shifts, valid_shifts_err)
    shift_mean_np = np.mean(valid_shifts)
    median = np.median(valid_shifts)

    # # Plot mean and err
    ax.vlines(shift_mean, -20, len(s) + 20, linestyle="dashed", alpha=0.5, color="black", label="Weighted average")
    ax.vlines(shift_mean_np, -20, len(s) + 20, linestyle="dashed", alpha=0.5, color="green", label="np.mean")
    ax.vlines(median, -20, len(s) + 20, linestyle="dashed", alpha=0.5, color="red", label="np.median")
    # ax.axvspan(shift_mean-shift_mean_err, shift_mean+shift_mean_err, alpha=0.2) # too small to see anyway

    # invert axis so feature 0 starts at the top
    plt.gca().invert_yaxis()

    if labels:
            ax.legend(bbox_to_anchor=(1.4, 0.99))
            ax.set_xlabel("Velocity Shift [m/s]")
            ax.set_ylabel("Feature")
            text = f'Weighted average shift = ({shift_mean:.3} ± {shift_mean_err:.1}) cm/s'
            ax.text(1.08, 0.84, text,
                    horizontalalignment='left',
                    verticalalignment='top',
                    transform=ax.transAxes)
    else:
            # hide ticks
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])


    # fig.savefig("fdsfds.png", dpi=400)



def plot_features_shift_matrix(result, coords, save_to_filename=None, do_not_show=False):
    size = np.max(np.max(coords)) + 1
    fig, axs = plt.subplots(size, size, figsize=(size, size))

    # Hide labels on all 
    for ax in axs.flat:
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])

    for x in np.arange(size):
        for y in np.arange(x, size):
            
            # Get index from coords
            try:
                index = list(coords).index((x, y))
                plot_features_shift(result[index], axs[x, y], labels=False)
            except:
                continue


    # Remove space between graphs
    plt.subplots_adjust(wspace=0, hspace=0)

    # Write column and row numbers
    cols = [col for col in range(0, size)]
    rows = [row for row in range(0, size)]
    for ax, col in zip(axs[0], cols):
        ax.set_title(col)
    for ax, row in zip(axs[:,0], rows):
        ax.set_ylabel(f"{row}     ", rotation=0, size='large')

    if save_to_filename != None:
        plt.savefig(f"figures/{save_to_filename}", dpi=100)

    if do_not_show:
        plt.close(fig) # don't show figure



def compute_matrix_multi_core(N_files = -1):
    """ doesn't work when put into function, idk why """
    import multiprocess

    N_processes = 6
    # N_features = 1000

    # Get list of files and find all features
    filenames = get_spectra_filenames_without_duplicate_dates()
    if N_files != -1:
        assert N_files > 0, "N_files is negative or zero"
        assert N_files <= len(filenames), "N_files is longer than number of data files"
        filenames = filenames[:N_files]

    features = []
    print("Finding features for all files...")
    for filename in tqdm(filenames):
        features.append(find_features(filename, log=False))

    # Setup coords
    size = len(filenames)
    # diff_matrix, diff_matrix_err, diff_matrix_valid = make_nan_matrix(size), make_nan_matrix(size), make_nan_matrix(size)

    # Compute one list of coords
    coords = []
    for x in np.arange(size):
        for y in np.arange(x, size):
            if x != y:
                coords.append((x, y)) 
            
            
    # Define function for each process
    def compute_shift_for_coords_chunk(coords):
        x = coords[0]
        y = coords[1]
        matches = find_feature_matches(features[x], features[y], log=False)
        shifts = compute_all_feature_shifts(matches, log=False) # TESTING: ONLY RUNNING SOME FEATURES
        return shifts


    if __name__ == '__main__':
        pool = multiprocess.Pool(processes = N_processes)
        # result = pool.map(compute_shift_for_coords_chunk, coords) # without tqdm
        
        # With progress bar
        result = []
        print("Computeing shifts for all files combinations...")
        for r in tqdm(pool.imap_unordered(compute_shift_for_coords_chunk, coords), total=len(coords)):
            result.append(r)

        return result, coords


def parse_matrix_results(result, coords):
    
    size = np.max(np.max(coords)) + 1
    diff_matrix, diff_matrix_err, diff_matrix_valid = make_nan_matrix(size), make_nan_matrix(size), make_nan_matrix(size)

    for coord, shifts in zip(coords, result):

        # Compute valid ratio, number of succesfull fits / total number of fits
        valids = shifts[:, 2]
        valid_ratio = len(valids[valids == 1])/len(valids)

        # Split 
        shifts_list, shifts_err_list, shifts_valid_list = shifts[:, 0], shifts[:, 1], shifts[:, 2]
        
        # Compute weighted average
        shift_mean, shift_mean_err = weighted_mean(shifts_list[shifts_valid_list == 1], shifts_err_list[[shifts_valid_list == 1]])
        median = np.median(shifts_list[shifts_valid_list == 1])

        x = coord[0]
        y = coord[1]

        # diff_matrix[x, y] = shift_mean
        diff_matrix[x, y] = median
        diff_matrix_err[x, y] = shift_mean_err
        diff_matrix_valid[x, y] = valid_ratio
        
    return diff_matrix, diff_matrix_err, diff_matrix_valid


def parse_matrix_results_fit(result, coords):
    """ Instead of taking weighted average, fit a straight line. This should
        give less attention to outliers. """
    
    size = np.max(np.max(coords)) + 1
    diff_matrix, diff_matrix_err, diff_matrix_valid = make_nan_matrix(size), make_nan_matrix(size), make_nan_matrix(size)

    for coord, shifts in zip(coords, result):

        # Compute valid ratio, number of succesfull fits / total number of fits
        valids = shifts[:, 2]
        valid_ratio = len(valids[valids == 1])/len(valids)

        # Split 
        shifts_list, shifts_err_list, shifts_valid_list = shifts[:, 0], shifts[:, 1], shifts[:, 2]
        
        # Compute weighted average
        shift_mean, shift_mean_err = weighted_mean(shifts_list[shifts_valid_list == 1], shifts_err_list[[shifts_valid_list == 1]])

        x = coord[0]
        y = coord[1]

        diff_matrix[x, y] = shift_mean
        diff_matrix_err[x, y] = shift_mean_err
        diff_matrix_valid[x, y] = valid_ratio
        
    return diff_matrix, diff_matrix_err, diff_matrix_valid


def fit_straight_line(x, y, y_err, init_value):
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(x, y, yerr=y_err, fmt=".", color="k")

    # Fitting functions:
    def func(x, a) :
        return a

    # ChiSquare fit model:
    def model_chi(a) :
        y_fit = func(x, a)
        chi2 = np.sum(((y - y_fit) / y_err)**2)
        return chi2
    model_chi.errordef = 1

    minuit = Minuit(model_chi, a=init_value)
    m = minuit.migrad()        
                        
    # Plot result
    xPeak = np.linspace(x[0], x[len(x)-1], 100)
    ax.plot(xPeak, func(xPeak, *minuit.values[:]), '-r')
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    Npoints = len(x)
    Nvar = 1                                        # Number of variables
    Ndof_fit = Npoints - Nvar                       # Number of degrees of freedom = Number of data points - Number of variables
    Chi2_fit = minuit.fval                          # The chi2 value
    Prob_fit = stats.chi2.sf(Chi2_fit, Ndof_fit)    # The chi2 probability given N degrees of freedom

    a, a_err = minuit.values['a'], minuit.errors['a']

    d = {'a':   [a, a_err],
        'Chi2':     Chi2_fit,
        'ndf':      Ndof_fit,
        'Prob':     Prob_fit,
        'f(x)=':    "a"
    }

    text = nice_string_output(d, extra_spacing=2, decimals=5)
    add_text_to_ax(0.62, 0.95, text, ax, fontsize=14, color='r')
    
    return a, m



def parse_matrix_results_df(result, coords):
    
    size = np.max(np.max(coords)) + 1
    diff_matrix, diff_matrix_err, diff_matrix_valid = make_nan_matrix(size), make_nan_matrix(size), make_nan_matrix(size)

    for coord, shifts in zip(coords, result):

        # Split 
        shift, shift_err, shift_valid = shifts[0], shifts[1], shifts[3]
    
        x = coord[0]
        y = coord[1]

        diff_matrix[x, y] = shift
        diff_matrix_err[x, y] = shift_err
        diff_matrix_valid[x, y] = shift_valid
        
    return diff_matrix, diff_matrix_err, diff_matrix_valid



def get_above_diagonal(matrix):
    """ returns returns the above diagonal in a mtrix """
    size = matrix.shape[0]
    off_diag = []
    for x in np.arange(0, size):
        if x + 1 < size:
            off_diag.append(matrix[x, x + 1])
    off_diag = np.asarray(off_diag)
    return off_diag


def get_time_interval_between_observations(dates):
    from datetime import datetime
    """ Returns list of date pairs, between which each observation is made """
    intervals = []
    first_day = datetime(*dates[0])
    for i in np.arange(len(dates) - 1):
        d1 = datetime(*dates[i])
        d2 = datetime(*dates[i+1])
        dd1 = d1 - first_day
        dd2 = d2 - first_day
        intervals.append([dd1.days, dd2.days])
    return np.asarray(intervals)
    

def compute_z_test_mask(x, sigma):
    """ Returns a mask with TRUEs if values are within the given number of 
    sigmas from the mean of the array """
    mean = np.mean(x)
    std = np.std(x)
    z = np.abs(x - mean)/std
    mask = (z < sigma)
    return mask


def filter_z_test(shifts, set_to_nan=False):
    """ return filtered shifts collection by z test of sigma 5 """
    
    # Filter
    df = pd.DataFrame(shifts, copy=True)
    df.columns = ["x", "err", "valid"]

    mask = compute_z_test_mask(df["x"], 5)

    if set_to_nan:
        df.valid[mask == False] = np.nan 
    else:
        df.valid[mask == False] = False 

    return np.asarray(df)



def filter_z_test_result(result, set_to_nan=False):
    """ Takes in the "result" object from the compute_all_feature_shifts and
    applies the z test filter. """
    filtered_result = []
    for file_result in result:
        s = filter_z_test(file_result, set_to_nan=set_to_nan)
        filtered_result.append(s)
    filtered_result = np.asarray(filtered_result, dtype=object)
    return filtered_result



def filter_IQR(shifts, set_to_nan=False):
    """ Filter shift results for outliers by IQR, taking only the ones that are within the 25-75 percentile.
        if set_to_nan is true, values outside will be set to np.nan, otherwise 0, marking them as invalid.

        https://www.askpython.com/python/examples/detection-removal-outliers-in-python
        https://stackoverflow.com/a/53338192/1692590
        """
    
    # Compute IQR range
    q25 = np.percentile(shifts[:, 0], 25, interpolation='midpoint')
    q75 = np.percentile(shifts[:, 0], 75, interpolation='midpoint')
    intr_qr = q75-q25
    vmin = q25-(1.5*intr_qr)
    vmax = q75+(1.5*intr_qr)

    # Filter
    df = pd.DataFrame(shifts, copy=True)
    df.columns = ["x", "err", "valid"]
    if set_to_nan:
        df.valid[df.x < vmin] = np.nan 
        df.valid[df.x > vmax] = np.nan
    else:
        df.valid[df.x < vmin] = 0 
        df.valid[df.x > vmax] = 0 

    return np.asarray(df)


def filter_IQR_result(result, set_to_nan=False):
    """ Takes in the "result" object from the compute_all_feature_shifts and
    applies the IQR filter. """
    filtered_result = []
    for file_result in result:
        s = filter_IQR(file_result, set_to_nan=set_to_nan)
        filtered_result.append(s)
    filtered_result = np.asarray(filtered_result, dtype=object)
    return filtered_result


def filter_IQR_dataframe_from_summed_diff(df, set_to_nan=False):
    """ Filter shift results for outliers by IQR, taking only the ones that are within the 25-75 percentile.
        if set_to_nan is true, values outside will be set to np.nan, otherwise 0, marking them as invalid.

        https://www.askpython.com/python/examples/detection-removal-outliers-in-python
        https://stackoverflow.com/a/53338192/1692590
    """
    
    # Compute IQR range
    q25 = np.percentile(df.summed_diff, 25, interpolation='midpoint')
    q75 = np.percentile(df.summed_diff, 75, interpolation='midpoint')
    intr_qr = q75-q25
    vmin = q25-(1.5*intr_qr)
    vmax = q75+(1.5*intr_qr)

    # Filter
    if set_to_nan:
        df.valid[df.summed_diff < vmin] = np.nan 
        df.valid[df.summed_diff > vmax] = np.nan
    else:
        df.valid[df.summed_diff < vmin] = False 
        df.valid[df.summed_diff > vmax] = False 
    
    return df


def compute_IQR_bounds(x):
    q25 = np.percentile(x, 25, interpolation='midpoint')
    q75 = np.percentile(x, 75, interpolation='midpoint')
    intr_qr = q75-q25
    vmin = q25-(1.5*intr_qr)
    vmax = q75+(1.5*intr_qr)
    return vmin, vmax



def matrix_reduce_results_file(filename, plot=True, input_is_angstrom=False):
    """ Takes a file of our cross-correlation results (matrix) and reduces"""
    result, coords = np.load(filename, allow_pickle=True)
    # result = filter_IQR_result(result)
    diff_matrix, diff_matrix_err, diff_matrix_valid = parse_matrix_results(result, coords)

    def model_chi2(*V):
        V = np.asarray([*V])
        res = []
        size = diff_matrix.shape[0] 
        # V = np.ones(size)
        for x in np.arange(size):
            # for y in np.arange(x, size - 1):
            for y in np.arange(x, size):
                if x != y:
                    diff_matrix[x, y]
                    V[x]
                    V[y]
                    res.append(((diff_matrix[x, y] - (V[x] - V[y])) / diff_matrix_err[x, y])**2)
        chi2 = np.sum(res)
        return chi2
    model_chi2.errordef = 1

    # Use the above diagonal as init values
    # init_values = get_above_diagonal(diff_matrix)
    init_values = np.zeros(diff_matrix.shape[0])

    minuit = Minuit(model_chi2, *init_values)
    m = minuit.migrad()
    final_shifts = minuit.values[:]
    final_shifts_err = minuit.errors[:]

    if plot == False:
        return m, final_shifts, final_shifts_err

    # Plot: 

    # The velocity shifts are between days, so let's put the x-error bar as the time span for each data point
    velocity_shifts = get_above_diagonal(diff_matrix)
    velocity_shifts_err = get_above_diagonal(diff_matrix_err)
    dates = get_spectra_dates(get_spectra_filenames_without_duplicate_dates())
    days = convert_dates_to_relative_days(dates)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 6))

    ax1.set_xlabel("Time [days]")
    ax1.set_ylabel("Velocity shift [cm/s]")
    ax1.set_title("Above diagonal")

    ax2.set_xlabel("Time [days]")
    ax2.set_title("Matrix chi2 reduction")

    # Plot above-diagonal
    for shift, shift_err, d in zip(velocity_shifts, velocity_shifts_err, days):
        if input_is_angstrom:
            shift = angstrom_to_velocity(shift)
        ax1.errorbar(d, shift, yerr=shift_err, fmt=".", color="k")

    # Plot matrix reduction results
    for shift, shift_err, d in zip(final_shifts, final_shifts_err, days):
        if input_is_angstrom:
            shift = angstrom_to_velocity(shift)
        ax2.errorbar(d, shift, yerr=shift_err, fmt=".", color="k")


    fig.tight_layout()
    # fig.savefig("rooo.png", dpi=300)
    return m, final_shifts, final_shifts_err


def matrix_reduce_results_rms(diff_matrix, plot=True, input_is_angstrom=False):

    def model_chi2(*V):
        V = np.asarray([*V])
        res = []
        size = diff_matrix.shape[0] 
        # V = np.ones(size)
        for x in np.arange(size):
            # for y in np.arange(x, size - 1):
            for y in np.arange(x, size):
                if x != y:
                    diff_matrix[x, y]
                    V[x]
                    V[y]
                    res.append(((diff_matrix[x, y] - (V[x] - V[y])) / diff_matrix[x, y] * 0.001 )**2)
        chi2 = np.sum(res)
        return chi2
    model_chi2.errordef = 1

    # Use the above diagonal as init values
    # init_values = get_above_diagonal(diff_matrix)
    init_values = np.zeros(diff_matrix.shape[0])

    minuit = Minuit(model_chi2, *init_values)
    m = minuit.migrad()
    final_shifts = minuit.values[:]
    final_shifts_err = minuit.errors[:]

    if plot == False:
        return m, final_shifts, final_shifts_err

    # Plot: 

    # The velocity shifts are between days, so let's put the x-error bar as the time span for each data point
    velocity_shifts = get_above_diagonal(diff_matrix)
    velocity_shifts_err = velocity_shifts * 0.001
    dates = get_spectra_dates(get_spectra_filenames_without_duplicate_dates())
    days = convert_dates_to_relative_days(dates)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 8))

    ax1.set_xlabel("Time [days]")
    ax1.set_ylabel("Velocity shift [cm/s]")
    ax1.set_title("Above diagonal")

    ax2.set_xlabel("Time [days]")
    ax2.set_title("Matrix chi2 reduction")

    # Plot above-diagonal
    for shift, shift_err, d in zip(velocity_shifts, velocity_shifts_err, days):
        if input_is_angstrom:
            shift = angstrom_to_velocity(shift)
        ax1.errorbar(d, shift, yerr=shift_err, fmt=".", color="k")

    # Plot matrix reduction results
    for shift, shift_err, d in zip(final_shifts, final_shifts_err, days):
        if input_is_angstrom:
            shift = angstrom_to_velocity(shift)
        ax2.errorbar(d, shift, yerr=0, fmt=".", color="k")


    fig.tight_layout()
    # fig.savefig("rooo.png", dpi=300)
    return m, final_shifts, final_shifts_err



def matrix_reduce_results_file_df(filename, plot=True):
    """ Takes a file of our cross-correlation results (matrix) and reduces"""
    result, coords = np.load(filename, allow_pickle=True)
    diff_matrix, diff_matrix_err, diff_matrix_valid = parse_matrix_results_df(result, coords)

    def model_chi2(*V):
        V = np.asarray([*V])
        res = []
        size = diff_matrix.shape[0] 
        # V = np.ones(size)
        for x in np.arange(size):
            for y in np.arange(x, size - 1):
                if x != y:
                    diff_matrix[x, y]
                    V[x]
                    V[y]
                    res.append(((diff_matrix[x, y] - (V[x] - V[y])) / diff_matrix_err[x, y])**2)
        chi2 = np.sum(res)
        return chi2
    model_chi2.errordef = 1

    # Use the above diagonal as init values
    init_values = get_above_diagonal(diff_matrix)

    minuit = Minuit(model_chi2, *init_values)
    m = minuit.migrad()
    final_shifts = minuit.values[:]
    final_shifts_err = minuit.errors[:]

    if plot == False:
        return m, final_shifts, final_shifts_err

    # Plot: 

    # The velocity shifts are between days, so let's put the x-error bar as the time span for each data point
    velocity_shifts = get_above_diagonal(diff_matrix)
    velocity_shifts_err = get_above_diagonal(diff_matrix_err)
    dates = get_spectra_dates(get_spectra_filenames_without_duplicate_dates())
    intervals = get_time_interval_between_observations(dates)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 8))

    ax1.set_xlabel("Time [days]")
    ax1.set_ylabel("Velocity shift [cm/s]")
    ax1.set_title("Above diagonal")

    ax2.set_xlabel("Time [days]")
    ax2.set_title("Matrix chi2 reduction")

    # Plot above diagonal
    for shift, shift_err, interval in zip(velocity_shifts, velocity_shifts_err, intervals):
        # print(shift, np.diff(interval), interval)
        days_interval = np.arange(*interval)
        ax1.plot(days_interval, [shift] * len(days_interval), linewidth=2, color="k")
        ax1.errorbar(np.median(days_interval), shift, yerr=shift_err, color="k") # add errorbar to the center point

    # Plot matrix reduction results
    for shift, shift_err, interval in zip(final_shifts, final_shifts_err, intervals):
        days_interval = np.arange(*interval)
        ax2.plot(days_interval, [shift] * len(days_interval), linewidth=2, color="k")
        ax2.errorbar(np.median(days_interval), shift, yerr=shift_err, color="k") # add errorbar to the center point

    fig.tight_layout()
    # fig.savefig("rooo.png", dpi=300)

    return m, final_shifts, final_shifts_err





def fit_final_shifts(final_shifts, final_shifts_err):

    dates = get_spectra_dates(get_spectra_filenames_without_duplicate_dates())
    # intervals = get_time_interval_between_observations(dates)
    # x = intervals[:, 0]
    x = convert_dates_to_relative_days(dates)
    y = final_shifts[:]
    y_err = final_shifts_err[:]

    fig, ax = plt.subplots(figsize=(16,8))
    ax.errorbar(x, y, yerr=y_err, fmt=".", color="k")


    # Fitting functions:
    def func(x, a, b, c, d) :
        return a * np.cos(x * b + c) + d


    # ChiSquare fit model:
    def model_chi(a, b, c, d) :
        y_fit = func(x, a, b, c, d)
        chi2 = np.sum(((y - y_fit) / y_err)**2)
        return chi2
    model_chi.errordef = 1

    minuit = Minuit(model_chi, a=2.5, b=1/50, c=1, d=1)
    m = minuit.migrad()        
                        
    # Plot result
    xPeak = np.linspace(x[0], x[len(x)-1], 100)
    ax.plot(xPeak, func(xPeak, *minuit.values[:]), '-r')
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Velocity shift [cm/s]")

    Npoints = len(x)
    Nvar = 4                                        # Number of variables
    Ndof_fit = Npoints - Nvar                       # Number of degrees of freedom = Number of data points - Number of variables
    Chi2_fit = minuit.fval                          # The chi2 value
    Prob_fit = stats.chi2.sf(Chi2_fit, Ndof_fit)    # The chi2 probability given N degrees of freedom

    a, a_err = minuit.values['a'], minuit.errors['a']
    b, b_err = minuit.values['b'], minuit.errors['b']
    c, c_err = minuit.values['c'], minuit.errors['c']
    d, d_err = minuit.values['d'], minuit.errors['d']
    wavel, wavel_err = (2 * np.pi)/b, np.sqrt((2 * np.pi)**2*b_err**2/b**4)

    d = {'A':   [a, a_err],
        'b':    [b, b_err],
        'c':    [c, c_err],
        'd':    [d, d_err],
        'λ':    [wavel, wavel_err],
        'Chi2':     Chi2_fit,
        'ndf':      Ndof_fit,
        'Prob':     Prob_fit,
        'f(x)=':     "A cos(bx + c) + d"
    }

    text = nice_string_output(d, extra_spacing=2, decimals=5)
    add_text_to_ax(0.62, 0.95, text, ax, fontsize=14, color='r')

    return a, b, c, d, a_err, b_err, c_err, d_err


# ================================================================================================
# 
#                                   Analysing order by order
# 
# ================================================================================================


def prepare_orders(filename, max_frac_err = 0.1, angstrom=False, is_peg_51 = False):
    """ Prepares data in a dataframe

    Data-filtering:
        - Data points outside EXCALIBUR_MASK are excluded.
        - Orders with less than 10 data points are excluded.
        - data points with large fractional error are excluded.

    Args:
        filename (fits file): the file
        max_frac_err (float, optional): exclude data points with larger fractional error (spectrum / photon count). Defaults to 0.1.
        angstrom (bool, optional): True if you want wavelengths in angstrom and not cm/s. Defaults to False.
        is_peg_51 (bool, optional) : True if using data from Peg51, which has other column names. Defaults to False.

    Returns:
        pd.DataFrame: Return a pandas dataframe from a given fits file with the following data:
            x : bary_excalibur in cm/s
            y : spectrum
            y_err : spectrum error
            order : order 
    """

    order_data = []
    fits_data = load_spectra_fits(filename)
    orders_n = shape(fits_data["spectrum"])[0]

    orders = np.arange(0, orders_n)
    orders = np.arange(37, 76)
    for order in orders:
        
        if is_peg_51:
            pixel_mask  = fits_data['pixel_mask'][order]
            y           = fits_data['spectrum'][order][pixel_mask]
            y_err       = fits_data['uncertainty'][order][pixel_mask]
            continuum   = fits_data['continuum'][order][pixel_mask]
            x           = fits_data['bary_wavelength'][order][pixel_mask]
        else: 
            excalibur_mask  = fits_data['EXCALIBUR_MASK'][order]
            y           = fits_data['spectrum'][order][excalibur_mask]
            y_err       = fits_data['uncertainty'][order][excalibur_mask]
            continuum   = fits_data['continuum'][order][excalibur_mask]
            x           = fits_data['BARY_EXCALIBUR'][order][excalibur_mask]
        
        # skip order if no good data
        if len(x) < 10 or len(y) < 10:
            continue

        # Normalize intensity by continuum 
        y = y/continuum
        y_err = y_err/continuum

        # Convert angstorm to cm/s
        if angstrom == False:
            x = angstrom_to_velocity(x)

        # filter by fractional error 
        frac_err = y_err/y
        frac_err_mask = (0 < frac_err) & (frac_err < max_frac_err) # reject if larger than 10% and negative
        y = y[frac_err_mask]
        y_err = y_err[frac_err_mask]
        x = x[frac_err_mask]

        # Now invert peaks
        y = 1 - y

        # Create DataFrame and append
        data = {
            'x': x,
            'y': y,
            'y_err': y_err,
            'order': order
            }
        order_data.append(pd.DataFrame(data))
    
    # concatenate dataframes
    df = pd.concat(order_data)
    return df



def compute_order_shift(df1, df2, order=-1, plot=False, zoom_plot=True, ax=None):
    """Computes the wavelength shift for a given order between two files (dataframes)

    Args:
        df1 (pd.dataframe): dataframe with order data for file 1
        df2 (pd.dataframe): dataframe with order data for file 2
        order (int, optional): Order number, just for the plot, or if needed in the returned dataframe. Defaults to -1.
        plot (bool, optional): Plot order. Defaults to False.
        zoom_plot (bool, optional): Zoom in on plot. Defaults to True.
        ax (plt.ax, optional): optional ax to plot on. Defaults to None.

    Returns:
        pd.dataframe: Dataframe with computed shift_val, shift_err, minuit_valid, summed_diff, order
    """

    # Add shift for self test 
    # df2.x = df2.x + 0.25
    
    # Interp first file
    f1 = interp1d(df1.x, df1.y, kind='cubic', fill_value="extrapolate")
    f1_upper_err = interp1d(df1.x, df1.y + df1.y_err, kind='cubic', fill_value="extrapolate")
    f1_lower_err = interp1d(df1.x, df1.y - df1.y_err, kind='cubic', fill_value="extrapolate")

    # ChiSquare fit model:
    def model_chi2(A):

        # Interpolate template
        interp_df2 = df2.x + A
        f2 = interp1d(interp_df2, df2.y, kind='cubic', fill_value="extrapolate")

        # Find common x-range
        xmin = max([min(df1.x), min(interp_df2)])
        xmax = min([max(df1.x), max(interp_df2)])
        xnewCommon = np.linspace(xmin, xmax, 1000)
        
        # Evaluate interpolation
        ynew1 = f1(xnewCommon)
        ynew2 = f2(xnewCommon)

        # Evalute error interpolation
        ynew1_upper_err = f1_upper_err(xnewCommon)
        ynew1_lower_err = f1_lower_err(xnewCommon)
        ynew1_errs = [np.abs(ynew1 - ynew1_upper_err), np.abs(ynew1 - ynew1_lower_err)] 
        ynew1_err = np.mean(ynew1_errs)

        # Compute chi2
        chi2 = np.sum(((ynew1 - ynew2) / ynew1_err)**2)
        return chi2
    model_chi2.errordef = 1
        
    A_init = 0
    minuit = Minuit(model_chi2, A=A_init)
    m = minuit.migrad()

    # Results
    valid = minuit.valid
    shift_min_final = minuit.values['A']
    shift_min_final_err = minuit.errors['A']
    summed_diff = np.sum(df1.y - df2.y) # doesn't have much to do with the fit, but just the difference from night to night in counts

    # Plot final shifted values
    if plot:

        if ax == None:
            fig, ax = plt.subplots(figsize=(14,6))

        ax.plot(df1.x, df1.y, ".", label=f"df1 ({order}. order)")
        ax.plot(df2.x, df2.y, ".",label="df2")
        ax.plot(df2.x + shift_min_final, df2.y, ".", label="Fit")
        ax.legend(loc="upper right")

        # Zoom:
        if zoom_plot:
            mid = np.median(f1.x)
            ax.set_xlim(mid - 2, mid + 2)
    

    # Create DataFrame and append
    data = {
        'shift_val':    [shift_min_final],
        'shift_err':    [shift_min_final_err],
        'minuit_valid': [valid],
        'summed_diff':  [summed_diff],
        'order':        [order]
        }
    return pd.DataFrame(data)


def compute_all_orders_shift(filename1, filename2, plot_overview=False, plot_orders=False, is_peg_51 = False):
    """Compute shift for all orders in two files.

    Args:
        filename1 (string path): fits file 1
        filename2 (string path): fits file 2
        plot_overview (bool, optional): plot the shifts for all orders in one plot. Defaults to False.
        plot_orders (bool, optional): plot all orders individually. Defaults to False.
        is_peg_51 (bool, optional) : True if using data from Peg51, which has other column names. Defaults to False.

    Returns:
        (float) : weighted mean for all orders
        (float) : weighted err for all orders
        (float) : ratio of valid minuit fits
        (float) : ratio of orders that passed IQR filter
    """
    

    f1 = prepare_orders(filename1, is_peg_51=is_peg_51)
    f2 = prepare_orders(filename2, is_peg_51=is_peg_51)

    # find common orders : interesection of lists of unique orders
    common_orders = np.intersect1d(np.unique(f1.order), np.unique(f2.order))

    results = []
    for o in common_orders:
        r = compute_order_shift(f1[f1.order == o], f2[f2.order == o], order=o, plot=plot_orders)    
        results.append(r)

    df = pd.concat(results)

    # Filter for outliers
    vmin, vmax = compute_IQR_bounds(df.shift_val)
    df["IQR_valid"] = (df.shift_val > vmin) & (df.shift_val < vmax)

    # Split to valid and invalid shifts
    df_valid = df[(df.IQR_valid == True) & (df.minuit_valid == True) ].copy()
    df_invalid = df[(df.IQR_valid == False) | (df.minuit_valid == False)].copy()

    # Compute weighted average
    shift, shift_err = weighted_mean(df_valid.shift_val, df_valid.shift_err)

    # plot
    if plot_overview:

        fig, ax1 = plt.subplots(figsize=(14, 10))
        ax1.errorbar(df_valid.shift_val, df_valid.order, xerr=df_valid.shift_err, fmt=".", linewidth=5, color = "C2")
        ax1.errorbar(df_invalid.shift_val, df_invalid.order, xerr=df_invalid.shift_err, fmt=".", linewidth=5, color = "C3")

        ax1.vlines(shift, min(df.order), max(df.order), linestyle="dashed", alpha=0.5, color="black", label=f"Weighted average = {shift:.3} ± {shift_err:.1}")
        ax1.axvspan(shift-shift_err, shift+shift_err, alpha=0.2) # too small to see anyway

        ax1.invert_yaxis()  # invert axis so feature 0 starts at the top
        ax1.set_title("Shift")
        ax1.set_xlabel("RV [cm/s]")
        ax1.set_ylabel("Order")
        ax1.legend()


    IQR_valid_ratio = len(df[df.IQR_valid]) / len(df)
    minuit_valid_ratio = len(df[df.minuit_valid]) / len(df)

    return shift, shift_err, minuit_valid_ratio, IQR_valid_ratio
