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
from calibration_functions import *
from tqdm import tqdm
import multiprocess
import pandas as pd

def angstrom_to_velocity(wavelength_shift):
    """ Converts wavelenth shift in angstrom to velocity shift in cm/s """
    c = 299792458
    angstrom_to_cm = 1e-8
    return wavelength_shift * angstrom_to_cm * c


def make_nan_matrix(size):
    matrix = np.empty((size,size))
    matrix[:] = np.nan
    return matrix


# Loads real spectra from EXPREES (Lily) for star HD34411
REAL_SPECTRA_path = "/Users/jakobharteg/Data/34411_spectra/"
def get_all_spectra_filenames():
    """ Returns all filenames of the spectra files in the path speficied in REAL_SPECTRA_path"""
    SPEKTRA_filenames = next(walk(REAL_SPECTRA_path), (None, None, []))[2]  # [] if no file
    SPEKTRA_filenames = sorted(SPEKTRA_filenames)
    return SPEKTRA_filenames

def load_spectra_fits(filename):
    """ Returns data from a fits file with a given name """
    path = REAL_SPECTRA_path  + "/" + filename
    hdul = fits.open(path)
    data = hdul[1].data.copy()
    hdul.close()
    return data

def get_spec_wavel(data, order, continuum_normalized=False, angstrom=False):
    """ Returns intensity, intensity_err and wavelength for a given spectra data.
        Use in conjunction with load_spectra_fits """

    data_spec       = data['spectrum'][order]
    data_spec_err   = data['uncertainty'][order]
    data_wavel      = data['wavelength'][order]

    if angstrom == False:
        data_wavel = angstrom_to_velocity(data_wavel) # convert angstrom to cm/s

    if continuum_normalized:
        cont = data['continuum'][order]
        data_spec = data_spec / cont
        data_spec_err = data_spec_err / cont

    return data_spec, data_spec_err, data_wavel


def load_spektra_date(filename):
    """ Returns the date of observation for a given fits filename """
    path = REAL_SPECTRA_path  + "/" + filename
    hdul = fits.open(path)
    header = hdul[0].header
    hdul.close()
    return header["DATE-OBS"]


def get_spectra_dates(filenames):
    """ Returns a list of dates (year, month, date) for given list of fits filenames """
    dates = []
    for i in np.arange(len(filenames)):
        date = load_spektra_date(filenames[i])
        date = date[:date.index(" ")]
        year, month, date = date.split("-")
        year, month, date = int(year), int(month), int(date)
        dates.append((year, month, date))
    return dates


# There are often several observations per night
def get_spectra_filenames_without_duplicate_dates():
    """ Returns a list of filesnames of the spectra files in the path speficied in REAL_SPECTRA_path without date-duplicates, i.e.
        oberservations taken on the same day. """
    all_files = get_all_spectra_filenames()
    all_dates = get_spectra_dates(all_files)
    files = [all_files[0]]
    dates = [all_dates[0]]
    for i in np.arange(1, len(all_dates)):
        if dates[-1] != all_dates[i]:
            dates.append(all_dates[i])
            files.append(all_files[i])
    return files


def plot_matrix(diff_matrix, diff_matrix_err, diff_matrix_valid):
    """ Plot shift matrices """

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(25,10))

    def fix_grid_lines(ax, size):
        ax.hlines(y=np.arange(0, size)+0.5, xmin=np.full(size, 0)-0.5, xmax=np.full(size, size)-0.5, color="black", alpha=0.2)
        ax.vlines(x=np.arange(0, size)+0.5, ymin=np.full(size, 0)-0.5, ymax=np.full(size, size)-0.5, color="black", alpha=0.2)


    # ======= PLOT 1 ============ Mean shifts
    cs = ax1.imshow(-diff_matrix)
    cax = make_axes_locatable(ax1).append_axes('right', size='5%', pad=0.05) # to make color bar same height at plot, needed when making several subplots with each colorbar
    cbar = fig.colorbar(cs, ax=ax1, cax=cax)
    fix_grid_lines(ax1, len(diff_matrix))

    # ======= PLOT 2 ============ Errors
    cs = ax2.imshow(diff_matrix_err)
    cax = make_axes_locatable(ax2).append_axes('right', size='5%', pad=0.05) # to make color bar same height at plot, needed when making several subplots with each colorbar
    cbar = fig.colorbar(cs, ax=ax2, cax=cax)
    fix_grid_lines(ax2, len(diff_matrix_err))

    # ======= PLOT 3 ============ Convergence ratio
    cs = ax3.imshow(diff_matrix_valid)
    cax = make_axes_locatable(ax3).append_axes('right', size='5%', pad=0.05) # to make color bar same height at plot, needed when making several subplots with each colorbar
    cbar = fig.colorbar(cs, ax=ax3, cax=cax)
    fix_grid_lines(ax3, len(diff_matrix_valid))

    # Fix spacing between plots
    fig.subplots_adjust(wspace=0.25)

    ax1.set_title("Shift factor")
    ax2.set_title("Error")
    ax3.set_title("Convergence ratio")



def find_features(filename, plot_orders = None, plot_features_in_order = None, log=True):
    """ Returns list of features x_values, y_values, y_err_values, x_peak_location, peak_index, order """

    # Filtering parameters
    max_frac_err        = 0.1       # maximum fractional error in intensity
    min_order_goodness  = 0.7       # Min fraction of data in an order that should be left after filtering for the order to be included. 
    min_peak_dist       = 50        # minimum distance (in pixels) between peaks  
    min_peak_prominence = 0.25      # minimum height of peak from base (not zero)
    
    feature_slices = []

    fits_data = load_spectra_fits(filename)
    orders_n = shape(fits_data["spectrum"])[0]

    orders = np.arange(0, orders_n)
    for order in orders:
        
        pixel_mask  = fits_data['pixel_mask'][order]    # filter by pixel mask
        y           = fits_data['spectrum'][order][pixel_mask]
        og_y        = y # copy of original y data before filtering
        y_err       = fits_data['uncertainty'][order][pixel_mask]
        continuum   = fits_data['continuum'][order][pixel_mask]
        x           = fits_data['wavelength'][order][pixel_mask]
    
        # Normalize intensity by continuum 
        y = y/continuum
        y_err = y_err/continuum

        # Convert angstorm to cm/s
        x = angstrom_to_velocity(x)

        # filter by fractional error 
        frac_err = y_err/y
        frac_err_mask = (0 < frac_err) & (frac_err < max_frac_err) # reject if larger than 10% and negative
        y = y[frac_err_mask]
        y_err = y_err[frac_err_mask]
        x = x[frac_err_mask]

        # Skip order if we filtered out more than 1 - min_order_goodness of the data (bad order ... )
        if len(y) / len(og_y) < min_order_goodness:
            continue

        # Now invert peaks
        y = 1 - y

        peaks = func_find_peaks(y, min_peak_dist, min_peak_prominence)
        peak_locs = peaks[0]
        peak_height = peaks[5] # peak height from y=0 
        
        # Plot
        if plot_orders is not None and (plot_orders == order).any():
            plt.figure(figsize=(30,3))
            plt.plot(x, y, ".")
            plt.plot(x[peak_locs], peak_height, "o", color="C3", label=f"{order}. order")
            plt.ylabel("1 - Continuum normalized counts")
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



def find_feature_matches(features1, features2, log=True, filter=True, relative_tolerance=0.0008, absolute_tolerance=5):
    """ Finds features in two files, and iterates from the lowest wavelength to the highest, in small
        steps, asking if either of the files have features close to the current wavelength within a
        defined tolerance. If so, it returns a match. 
    """

    peaks1 = np.array(features1[:, 3], dtype=float)
    peaks2 = np.array(features2[:, 3], dtype=float)

    max1, max2 = np.max(peaks1), np.max(peaks2)
    min1, min2 = np.min(peaks1), np.min(peaks2)

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
                f1_peak, f2_peak = f1[3], f2[3]
                if abs(f1_peak - f2_peak) > 5:
                    continue
                    
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



def compute_feature_shift(x1, y1, y1_err, peak1, x2, y2, peak2, plot=False, ax=None):
    """ Attempts to fit two features with based on a shift parameter.
        Returns shift_min_final, shift_min_final_err, valid """
    
    # Add shift for self test 
    # x2 = x2 + 0.25
    
    # Interp first file
    f1 = interp1d(x1, y1, kind='cubic', fill_value="extrapolate")
    f1_upper_err = interp1d(x1, y1 + y1_err, kind='cubic', fill_value="extrapolate")
    f1_lower_err = interp1d(x1, y1 - y1_err, kind='cubic', fill_value="extrapolate")

    # ChiSquare fit model:
    def model_chi2(A):

        # Interpolate template
        interp_x2 = x2 + A
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
        ynew1_errs = [np.abs(ynew1 - ynew1_upper_err), np.abs(ynew1 - ynew1_lower_err)] 
        ynew1_err = np.mean(ynew1_errs)

        # Compute chi2
        chi2 = np.sum(((ynew1 - ynew2) / ynew1_err)**2)
        return chi2
    model_chi2.errordef = 1
        
    A_init = peak1 - peak2
    minuit = Minuit(model_chi2, A=A_init)
    minuit.migrad()

    # Results
    valid = minuit.valid
    shift_min_final = minuit.values['A']
    shift_min_final_err = minuit.errors['A']

    # Plot final shifted values
    if plot:

        if ax == None:
            fig, ax = plt.subplots(figsize=(14,6))

        ax.plot(x1, y1)
        ax.plot(x2 + shift_min_final, y2)
        # ax.plot(x1, y2 - y1, "k")
        
        # summed_diff = np.sum(x1 - (x2 + shift_min_final))
        # if summed_diff > diff:
        #     ax.set_facecolor('honeydew')

        if np.isclose(np.sum(y1), np.sum(y2), 0.1):
            ax.set_facecolor('honeydew')


    return shift_min_final, shift_min_final_err, valid


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


def plot_features_shift(shifts, ax=None, labels=True):

    s, s_err, s_valid = shifts[:, 0], shifts[:, 1], shifts[:, 2]

    feature_n = np.arange(len(s))
    valid_shifts, valid_shifts_err, valid_features          = s[s_valid == 1], s_err[s_valid == 1], feature_n[s_valid == 1]
    invalid_shifts, invalid_shifts_err, invalid_features    = s[s_valid == 0], s_err[s_valid == 0], feature_n[s_valid == 0]

    if ax == None:
            fig, ax = plt.subplots(figsize=(16, 20))

    # ax.errorbar(valid_shifts, valid_features, xerr=valid_shifts_err * 100, fmt="none", linewidth=1, color = "C2", label=rf'{len(valid_shifts)} valid fits, with errors $\times 100$')
    # ax.errorbar(invalid_shifts, invalid_features, xerr=invalid_shifts_err * 100, fmt="none", linewidth=1, color = "C3", label=rf'{len(invalid_shifts)} invalid fits, with errors $\times 100$')
    ax.errorbar(valid_shifts, valid_features, xerr=valid_shifts_err, fmt="none", linewidth=1, color = "C2", label=rf'{len(valid_shifts)} valid fits, with errors')
    ax.errorbar(invalid_shifts, invalid_features, xerr=invalid_shifts_err, fmt="none", linewidth=1, color = "C3", label=rf'{len(invalid_shifts)} invalid fits, with errors')

    ax.scatter(valid_shifts, valid_features)
    ax.scatter(invalid_shifts, invalid_features)


    shift_mean, shift_mean_err = weighted_mean(valid_shifts, valid_shifts_err)

    # # Plot mean and err
    ax.vlines(shift_mean, 0, len(s), linestyle="dashed", alpha=0.5, color="black", label="Weighted average")
    # ax.axvspan(shift_mean-shift_mean_err, shift_mean+shift_mean_err, alpha=0.2) # too small to see anyway

    # invert axis so feature 0 starts at the top
    plt.gca().invert_yaxis()

    if labels:
            ax.legend(bbox_to_anchor=(1.4, 0.99))
            ax.set_xlabel("Velocity Shift [cm/s]")
            ax.set_ylabel("Feature")
            ax.text(1.08, 0.89, f'Weighted average shift = ({shift_mean:.3} ± {shift_mean_err:.1}) cm/s',
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


def compute_matrix_multi_core(N_files = -1):
    N_processes = 6
    # N_features = 1000

    # Get list of files and find all features
    filenames = get_spectra_filenames_without_duplicate_dates()
    if N_files != -1:
        assert N_files > 0, "N_files is negative or zero"
        assert N_files <= len(filenames), "N_files is longer than number of data files"
        filenames = filenames[:N_files]

    features = [find_features(filename, log=False) for filename in filenames]

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


    # if __name__ == '__main__':
    
    pool = multiprocess.Pool(processes = N_processes)
    # result = pool.map(compute_shift_for_coords_chunk, coords) # without tqdm
    
    # With progress bar
    result = []
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

        # Filter away larger values by setting valid to zero (false)
        # df = pd.DataFrame(shifts, copy=True)
        # df.columns = ["x", "err", "valid"]
        # df.valid[:] = 1 # reset ... 
        # df.valid[df.x > 0.25] = 0 
        # df.valid[df.x < -0.25] = 0 
        # shifts = np.asarray(df)

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
    


def filter_IQR(shifts, set_to_nan=False):
    """ Filter shift results for outliers by IQR, taking only the ones that are within the 25-75 percentile.
        if set_to_nan is true, values outside will be set to np.nan, otherwise 0, marking them as invalid. 
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


