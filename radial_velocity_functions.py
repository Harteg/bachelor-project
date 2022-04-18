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
import pandas as pd

sys.path.append('/Users/jakobharteg/Github/MyAppStat/')
from ExternalFunctions import nice_string_output, add_text_to_ax # useful functions to print fit results on figure


def angstrom_to_velocity(wavelength_shift):
    """ Converts wavelenth shift in angstrom to velocity shift in cm/s """
    c = 299792458
    angstrom_to_cm = 1e-8
    return wavelength_shift * angstrom_to_cm * c


def velocity_to_angstrom(velocity):
    """ Converts velocity in cm/s to wavelength in angstrom"""
    c = 299792458
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

def get_all_spectra_filenames(spectra_path = SPECTRA_PATH_34411):
    """ Returns all filenames of the spectra files in the path speficied in spectra_path"""
    SPEKTRA_filenames = next(walk(spectra_path), (None, None, []))[2]  # [] if no file
    SPEKTRA_filenames = sorted(SPEKTRA_filenames)
    return SPEKTRA_filenames

def load_spectra_fits(filename, spectra_path = SPECTRA_PATH_34411):
    """ Returns data from a fits file with a given name """
    path = spectra_path  + "/" + filename
    hdul = fits.open(path)
    data = hdul[1].data.copy()
    hdul.close()
    return data

def get_spec_wavel(data, order, continuum_normalized=False, angstrom=False):
    """ Returns intensity, intensity_err and wavelength for a given spectra data.
        Use in conjunction with load_spectra_fits """

    data_spec       = data['spectrum'][order]
    data_spec_err   = data['uncertainty'][order]
    # data_wavel      = data['wavelength'][order]
    data_wavel      = data['BARY_EXCALIBUR'][order]

    if angstrom == False:
        data_wavel = angstrom_to_velocity(data_wavel) # convert angstrom to cm/s

    if continuum_normalized:
        cont = data['continuum'][order]
        data_spec = data_spec / cont
        data_spec_err = data_spec_err / cont

    return data_spec, data_spec_err, data_wavel


def get_spektra_date(filename, spectra_path = SPECTRA_PATH_34411):
    """ Returns the date of observation for a given fits filename """
    path = spectra_path  + "/" + filename
    hdul = fits.open(path)
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


# There are often several observations per night
def get_spectra_filenames_without_duplicate_dates(spectra_path = SPECTRA_PATH_34411):
    """ Returns a list of filesnames of the spectra files in the path speficied in SPECTRA_PATH_34411 without date-duplicates, i.e.
        oberservations taken on the same day. """
    all_files = get_all_spectra_filenames(spectra_path)
    all_dates = get_spectra_dates(all_files, spectra_path)
    files = [all_files[0]]
    dates = [all_dates[0]]
    for i in np.arange(1, len(all_dates)):
        if dates[-1] != all_dates[i]:
            dates.append(all_dates[i])
            files.append(all_files[i])
    return files


def plot_spectra_dates(spectra_dates):
    from datetime import datetime
    plt.figure(figsize=(25, 8))
    for index, date in enumerate(spectra_dates):
        year, month, date = date
        d = datetime(year, month, date)
        plt.scatter(d, index, color="k", s=2)


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
    cbar.set_label('cm/s', rotation=270)
    fix_grid_lines(ax1, len(diff_matrix))

    # ======= PLOT 2 ============ Errors
    cs = ax2.imshow(diff_matrix_err)
    cax = make_axes_locatable(ax2).append_axes('right', size='5%', pad=0.05) # to make color bar same height at plot, needed when making several subplots with each colorbar
    cbar = fig.colorbar(cs, ax=ax2, cax=cax)
    fix_grid_lines(ax2, len(diff_matrix_err))

    # ======= PLOT 3 ============ Valid/Convergence ratio
    cs = ax3.imshow(diff_matrix_valid)
    cax = make_axes_locatable(ax3).append_axes('right', size='5%', pad=0.05) # to make color bar same height at plot, needed when making several subplots with each colorbar
    cbar = fig.colorbar(cs, ax=ax3, cax=cax)
    fix_grid_lines(ax3, len(diff_matrix_valid))

    # Fix spacing between plots
    fig.subplots_adjust(wspace=0.25)

    ax1.set_title("Radial velocity shift")
    ax2.set_title("Error")
    ax3.set_title("Valid ratio")



# def find_features(filename, plot_orders = None, plot_features_in_order = None, log=True):
def find_features(filename, 
                    plot_orders = None, 
                    plot_features_in_order = None, 
                    log=True, 
                    max_frac_err = 0.1,                 # maximum fractional error in intensity
                    min_order_goodness = 0.7,           # Min fraction of data in an order that should be left after filtering for the order to be included. 
                    min_peak_dist = 50,                 # minimum distance (in pixels) between peaks  
                    min_peak_prominence = 0.25          # minimum height of peak from base (not zero)
    ):
    
    """ Returns list of features x_values, y_values, y_err_values, x_peak_location, peak_index, order """

    
    feature_slices = []
    fits_data = load_spectra_fits(filename)
    orders_n = shape(fits_data["spectrum"])[0]

    orders = np.arange(0, orders_n)
    for order in orders:
        
        # pixel_mask  = fits_data['pixel_mask'][order]    # filter by pixel mask
        excalibur_mask  = fits_data['EXCALIBUR_MASK'][order]    # filter by EXCALIBUR_MASK
        y           = fits_data['spectrum'][order][excalibur_mask]
        og_y        = y # copy of original y data before filtering
        y_err       = fits_data['uncertainty'][order][excalibur_mask]
        continuum   = fits_data['continuum'][order][excalibur_mask]
        # x           = fits_data['wavelength'][order][excalibur_mask]
        x           = fits_data['BARY_EXCALIBUR'][order][excalibur_mask]

        # skip order if no good data
        if len(x) == 0 or len(y) == 0:
            continue

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
            # print("Order sucks:", len(y), "/", len(og_y), "=", len(y) / len(og_y))
            continue

        # Now invert peaks
        y = 1 - y

        peaks = func_find_peaks(y, min_peak_dist, min_peak_prominence)
        peak_locs = peaks[0]
        peak_height = peaks[5] # peak height from y=0 
        
        # Plot
        if plot_orders is not None and (plot_orders == order).any():
            plt.figure(figsize=(30,3))
            plt.plot(velocity_to_angstrom(x), y, ".")
            plt.plot(velocity_to_angstrom(x[peak_locs]), peak_height, "o", color="C3", label=f"{order}. order")
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



def compute_feature_shift(x1, y1, y1_err, peak1, x2, y2, peak2, plot=False, ax=None):
    """ Attempts to fit two features with based on a shift parameter.
        Returns shift_min_final, shift_min_final_err, valid """
    
    # Add shift for self test 
    # x2 = x2 + 0.25
    
    # if len(x1) == 0 or len(y1) == 0 or len(x2) == 0 or len(y2) == 0:
    #     return 0, 0, 0

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



def plot_features_shift_matrix(result, coords, save_to_filename=None, do_not_show=False):
    size = np.max(np.max(coords)) + 1
    fig, axs = plt.subplots(size, size, figsize=(size*2, size*2))

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


def matrix_reduce_results_file(filename, plot=True):
    """ Takes a file of our cross-correlation results (matrix) and reduces"""
    result, coords = np.load(filename, allow_pickle=True)
    result = filter_IQR_result(result)
    diff_matrix, diff_matrix_err, diff_matrix_valid = parse_matrix_results(result, coords)

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
    dates = get_spectra_dates(get_spectra_filenames_without_duplicate_dates())
    intervals = get_time_interval_between_observations(dates)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 8))

    ax1.set_xlabel("Time [days]")
    ax1.set_ylabel("Velocity shift [cm/s]")
    ax1.set_title("Above diagonal")

    ax2.set_xlabel("Time [days]")
    ax2.set_title("Matrix chi2 reduction")

    # Plot above diagonal
    for shift, interval in zip(velocity_shifts, intervals):
        # print(shift, np.diff(interval), interval)
        days_interval = np.arange(*interval)
        ax1.plot(days_interval, [shift] * len(days_interval), linewidth=2)

    # Plot matrix reduction results
    for shift, interval in zip(final_shifts, intervals):
        days_interval = np.arange(*interval)
        ax2.plot(days_interval, [shift] * len(days_interval), linewidth=2)

    fig.tight_layout()
    # fig.savefig("rooo.png", dpi=300)

    return m, final_shifts, final_shifts_err



def fit_final_shifts(final_shifts, final_shifts_err):

    dates = get_spectra_dates(get_spectra_filenames_without_duplicate_dates())
    intervals = get_time_interval_between_observations(dates)
    x = intervals[:, 0]
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

