import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import cartopy
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker
import shapely
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 
import imageio
import os
import sklearn.metrics as metrics
import xarray as xr

"""
Reproducible code for "A machine learning emulator for Lagrangian particle dispersion model footprints: A case study using NAME" 
by Elena Fillola, Raul Santos-Rodriguez, Alistair Manning, Simon O'Doherty and Matt Rigby (2022)

Author: Elena Fillola (elena.fillolamayoral@bristol.ac.uk)
"""


class MakePredictions:
    """
    Make footprint predictions

    inputs:
        - clfs: list of regressors, of the same size as data.size**2
        - data: LoadData object. data.site should match info["site"] (ie the site the regressors were trained on). 
            Otherwise, predictions will be produced but of low quality
        - inputs: np array with meteorological inputs, extracted from LoadData object. Should have shape (n_samples, n_inputs*(metsize**2))
        - jump: parameter passed to get_all_inputs
    outputs:
        MakePredictions object with attributes:
        - truths: np array of footprints, identical to data.fp_data[jump:-3,:]
        - predictions: np array of emulated footprints, of same shape as truths
        - inputs, clfs, size, metsize, jump: parameters and info from inputs

    Further functions:
        - save_predictions: save emulated footprints as a .nc file
        - predict_fluxes: convolute footprints with fluxes to obtain above-baseline mole fraction in area
        - plot_flux: plots the above-baseline mole fractions for a particular month
        - plot_footprint: plots a side-by-side comparison of a footprint at a particular point in time
        - make_footprint_gif: creates a gif of footprints side-by-side between two dates
    """
    def __init__(self, clfs, data, inputs, jump):
        self.inputs = inputs
        self.clfs = clfs
        self.data = data
        
        ## set up prediction arrays and other attributes
        self.truths = data.fp_data[jump:-3,:]
        self.predictions = np.empty_like(self.truths)
        
        self.size = data.size
        self.metsize = data.metsize
        self.jump = jump

        sizediff = int((self.metsize-self.size)/2)
        centre = int(self.size/2)

        ## around measurement indeces (fixed for all regressors)
        around_measurement_indeces = [(centre+sizediff,j+centre+sizediff) for j in [0,-1,1]] + [(j + centre+sizediff, centre+sizediff) for j in [-1,1]] + [(i+centre+sizediff, j+centre+sizediff) for (i,j) in [(-1,-1), (+1,+1), (+1,-1), (-1,+1)]]
                
        ## for each regressor, set up indeces, predict and store
        for tree in range(len(clfs)):
            self.predictions[:,tree] = predict_tree(self.clfs[tree], self.data, self.inputs, tree, return_truths=False)
        

    def save_predictions(self, info, save_dir=None, year=None):
        """
        save emulated footprints as a .nc file

        Takes as inputs the trained model info dict, a year as an int (if the LoadData object contains more than year) and the directory and name format to save the footprints. The year, month and extension (.nc) will be added when saving (so pass save_dir in format "/path/to/folder/footprint_format_name")

        """
        if save_dir == None:
            save_dir = "data/emulated_fps/"+self.data.site+"_"

        if year==None:
            if type(self.data.year) == int:
                year = self.data.year
            else:
                print("No year was passed but the LoadData object comprises more than one year. Please pass a year.")
        assert type(year) == int, "Year should be an integer"


        months = np.unique(pd.DatetimeIndex(self.data.met.time.values[self.jump:-3][(pd.DatetimeIndex(self.data.met.time.values[self.jump:-3]).year == year) ]).month) 
        for m in months:
            month = np.argwhere((pd.DatetimeIndex(self.data.met.time.values[self.jump:-3]).month == m) & (pd.DatetimeIndex(self.data.met.time.values[self.jump:-3]).year == year)).flatten()

            to_save = np.reshape(self.predictions[month, :], (len(month), self.size, self.size))
            to_save = np.transpose(to_save, [1,2,0])
            data_var = {'fp':(['lat', 'lon', 'time'], to_save, self.data.fp_data_full.fp.attrs)}
            coords = {'lat':(['lat'], self.data.fp_lats), 'lon':(['lon'], self.data.fp_lons), 'time':(['time'], self.data.met.time.values[self.jump:-3][month])}
            attrs = {'emulated_with':'footprints_emulator at https://github.com/elenafillo/LPDM-emulation-trees', 'emulation_date': str(datetime.now()), 'original_fp_attrs': str(self.data.fp_data_full.attrs), 'emulator_info': str(info), 'regressor_params' : str(self.clfs[0].get_params())} 

            monthly_fp = xr.Dataset(data_vars=data_var, coords=coords, attrs=attrs)

            save_path = save_dir + str(year) + "{:02d}.nc".format(m)

            monthly_fp.to_netcdf(save_path)



    def predict_fluxes(self, flux, units_transform = "default"):
        """
        convolute predicted footprints and fluxes.
        Returns two np arrays, one with the true flux and one with the emulated flux, both of shape (n_footprints,)
        input flux should be an array, regridded and cut to the same resolution and size of the footprints
        units_transform can be None (use fluxes directly), "default" (performs flux*1e3 / CH4molarmass) or another function (which should return an array of the same shape as the original flux)
        """
        shape = self.size 
        if units_transform != None:
            if units_transform == "default":
                molarmass = 16.0425
                flux = flux*1e3 / molarmass
            else:
                flux = units_transform(flux)
        true_concentration = np.reshape(self.truths, (len(self.truths), shape, shape))*flux
        self.true_flux = np.sum(true_concentration, axis = (1,2))
        pred_concentration = np.reshape(self.predictions, (len(self.predictions), shape, shape))*flux
        self.pred_flux = np.sum(pred_concentration, axis = (1,2))
        
        return self.true_flux, self.pred_flux


    def plot_flux(self, month, year=None):
        """
        Plot mole fraction from LPDM footprints and emulated footprints for a particular month.

        Takes as inputs the month to plot (and year if data has more than one year), as an integer (January is 1, February 2 etc and 2015, 2016 etc). The year is not necessary if the LoadData object contains only one year. Note that the function predict_fluxes needs to have been run to be able to plut the fluxes.

        """
        fig, axis = plt.subplots(1,1,figsize = (15,4))

        fontsizes = {"title":17, "labels": 15, "axis":10}

        if year==None:
            if type(self.data.year) == int:
                year = self.data.year
            else:
                print("No year was passed but the LoadData object comprises more than one year. Please pass a year.")

        month = np.argwhere((pd.DatetimeIndex(self.data.met.time.values[self.jump:-3]).month == month) & (pd.DatetimeIndex(self.data.met.time.values[self.jump:-3]).year == year))
        assert len(month) > 0, "It seems there is no data in the LoadData object for this month and year."
        dates = pd.DatetimeIndex(self.data.met.time.values[self.jump:-3][month])
        month_idxs = [month[0][0],month[-1][0]+1]
        
        try:
            axis.plot(dates, 1e6*self.true_flux[month_idxs[0]:month_idxs[1]], label = "using LPDM-generated footprints", linewidth=2 ,c="#2c6dae")
            axis.plot(dates, 1e6*self.pred_flux[month_idxs[0]:month_idxs[1]], label = "using emulated footprints", linewidth=2 ,c="#989933")
        except AttributeError:
            print("MakePredictions object needs true_flux and pred_flux attributes. Run the predict_fluxes function before plotting.")
        
        axis.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.WE))
        axis.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
        axis.xaxis.set_minor_locator(mdates.DayLocator(bymonthday=range(1, 32)))

        axis.yaxis.offsetText.set_fontsize(0)
        
        axis.set_ylim([0,0.0000001+1e6*np.max([self.pred_flux[month_idxs[0]:month_idxs[1]], self.true_flux[month_idxs[0]:month_idxs[1]]])])

        axis.set_ylabel('Above baseline methane concentration, (micro mol/mol)', fontsize=fontsizes["axis"])
        axis.set_title('Above baseline methane concentration for ' + self.data.site)

        axis.legend()

        plt.show()

    def plot_footprint(self, date, fixed_cbar = False):
        """
        Plot footprints (real and predicted side-by-side).

        Takes as inputs the date to plot (as a string in format hour:00 day/month/year (eg 15:00 1/3/2016) or as an data index as an int).
        """
        ## check that date is within range, and get index from date (or viceversa)
        if type(date) == str:
            try:
                date = datetime.strptime(date, "%H:00 %d/%m/%Y")
            except:
                print("Something was wrong with the date. Please pass it in format hour:00 day/month/year (eg 15:00 1/3/2016)")
            
            try:
                idx = pd.DatetimeIndex(self.data.met.time.values[self.jump:-3]).get_loc(date)
            except:
                raise KeyError("This date is out of range")

        if type(date) == int:
            idx = date
            try:
                date = pd.DatetimeIndex(self.data.met.time.values[self.jump:-3])[idx]
            except:
                raise KeyError("This index is out of range")

        ## create figure and plot
        fig, (axr, axp) = plt.subplots(1,2,figsize = (15,7), subplot_kw={'projection':cartopy.crs.Mercator()})

        if fixed_cbar==False:
            vmax = np.nanmax(self.truths[idx,:])
        else:
            vmax = fixed_cbar
        
        axr.pcolormesh(self.data.fp_lons, self.data.fp_lats, np.reshape(self.truths[idx,:], (10,10)), transform=cartopy.crs.PlateCarree(), cmap="Reds", vmax = vmax, vmin=0)
        c = axp.pcolormesh(self.data.fp_lons, self.data.fp_lats, np.reshape(self.predictions[idx,:], (10,10)), transform=cartopy.crs.PlateCarree(), cmap="Reds", vmax = vmax, vmin=0)


        ## set up axis
        for ax in [axr, axp]:
            ax.set_extent([self.data.fp_lons[0]-0.1,self.data.fp_lons[-1]+0.1, self.data.fp_lats[0]+0.1,self.data.fp_lats[-1]+0.1], crs=cartopy.crs.PlateCarree())
            ax.set_xticks(self.data.fp_lons[::3], crs=cartopy.crs.PlateCarree())
        
            lon_formatter = LongitudeFormatter(number_format='.1f', degree_symbol='', dateline_direction_label=True)
            ax.xaxis.set_major_formatter(lon_formatter)  
            ax.set_yticks(self.data.fp_lats[::3], crs=cartopy.crs.PlateCarree())
            lat_formatter = LatitudeFormatter(number_format='.1f',  degree_symbol='',)
            ax.yaxis.set_major_formatter(lat_formatter)             
            ax.tick_params(axis='both', which='major', labelsize=12)   

            ax.plot(self.data.release_lon+0, self.data.release_lat+0, marker='o', c="w", markeredgecolor = "k", transform=cartopy.crs.PlateCarree(), markersize=5)

            ax.coastlines(resolution='50m', color='black', linewidth=2)

        
        axr.set_title("LPDM-generated footprint - "+ self.data.site + "\n" + date.strftime("%m/%d/%Y, %H:00"), fontsize = 17)
        axp.set_title("Emulator-generated footprint - "+ self.data.site + "\n" + date.strftime("%m/%d/%Y, %H:00"), fontsize = 17)

        ## set up cbar
        cbar = plt.colorbar(c, ax=[axr, axp], orientation="vertical", aspect = 15, pad = 0.02)
        cbar.ax.tick_params(labelsize=11)
        cbar.set_label("sensitivity, (mol/mol)/(mol/m2/s)", size = 15, loc="center", labelpad = 16) 

        fig.show()


    def make_footprint_gif(self, start_date, end_date, savepath=None):
        """
        Make gif of footprints (real and predicted side-by-side) for a range of dates.

        Takes as inputs the start and end date (as strings in format hour:00 day/month/year (eg 15:00 1/3/2016) or as integers), and an optional saving directory. Otherwise gif is saved in local directory under the name footprints_startdate_enddate.gif.

        """
        
        ## check that date is within range, and get index from date (or viceversa)
        if type(start_date) == str and type(end_date) == str:
            try:
                start_date = datetime.strptime(start_date, "%H:00 %d/%m/%Y")
                end_date = datetime.strptime(end_date, "%H:00 %d/%m/%Y")
            except:
                print("Something was wrong with the date. Please pass it in format hour:00 day/month/year (eg 15:00 1/3/2016)")
            
            try:
                start_idx = pd.DatetimeIndex(self.data.met.time.values[self.jump:-3]).get_loc(start_date)
                end_idx = pd.DatetimeIndex(self.data.met.time.values[self.jump:-3]).get_loc(end_date)
            except KeyError:
                raise KeyError("One of the dates is out of range")

            assert start_idx<end_idx, "Start date should be before end date"

        if type(start_date) == int and type(end_date) == int:
            start_idx, end_idx = start_date, end_date 
            assert start_idx<end_idx, "Start date should be before end date"
            try:
                start_date = pd.DatetimeIndex(self.data.met.time.values[self.jump:-3])[start_idx]
                end_date = pd.DatetimeIndex(self.data.met.time.values[self.jump:-3])[end_idx]
            except KeyError:
                raise KeyError("One of the dates is out of range")

        filenames = []

        vmax = 0.5*np.nanmax(self.truths[start_idx:end_idx,:])

        if savepath==None:
            savepath="footprints_"+ start_date.strftime("%H-%d-%m-%Y") + "_" + end_date.strftime("%H-%d-%m-%Y") + ".gif"
            print("saving as ", savepath )

        # plot each figure and save
        for t in range(start_idx, end_idx):
            self.plot_footprint(t, fixed_cbar=vmax)
            filename = f'{t}.png'
            filenames.append(filename)       
            plt.savefig(filename)
            plt.close()

        # write gif
        try:
            with imageio.get_writer(savepath, mode='I') as writer:
                for filename in filenames:
                    image = imageio.imread(filename)
                    writer.append_data(image)
        except Exception as inst:
            print("Gif could not be saved ", inst)
        # Remove files
        for filename in set(filenames):
            os.remove(filename)


    ### metrics


    def NMAE_fp(self):
        # Returns Normalised Mean Absolute Error for the footprints
        return (metrics.mean_absolute_error(self.truths, self.predictions)/(np.mean(self.truths)))

    def above_threshold_acc(self, threshold=0):
        # Returns percentage of cells correctly predicted to be below/above a threshold
        mask = self.truths > threshold
        mask_pred = self.predictions > threshold       
        return 100*np.sum(mask == mask_pred)/(np.shape(mask)[0]*np.shape(mask)[1])

    def NMAE_flux(self):
        # Returns Normalised Mean Absolute Error for the flux
        try:
            NMAE = metrics.mean_absolute_error(self.true_flux, self.pred_flux)/(np.mean(self.true_flux))
        except AttributeError:
            print("Flux metrics need true_flux and pred_flux attributes. Run the predict_fluxes function before evaluating.")
        return NMAE


    def R_squared(self):
        # Returns R2 score for the flux
        try:
            R2 = metrics.r2_score(self.true_flux, self.pred_flux)
        except AttributeError:
            print("Flux metrics need true_flux and pred_flux attributes. Run the predict_fluxes function before evaluating.")
        return R2

    def bias(self):
        # Returns Mean Bias Error
        try:
            bias = np.mean(self.pred_flux - self.true_flux)
        except AttributeError:
            print("Flux metrics need true_flux and pred_flux attributes. Run the predict_fluxes function before evaluating.")

        return bias
    
    def plot_bias(self, quantiles=10):
        # Plots the bias for the flux when divided into q quantiles
        try:
            bias = []
            qs = [np.quantile(self.pred_flux, n) for n in np.linspace(0, 1, quantiles+1)]
            for q in range(len(qs)-1):
                mask = np.bitwise_and(self.true_flux>qs[q], self.true_flux<qs[q+1])
                bias.append(np.mean(self.pred_flux[mask]-self.true_flux[mask]))    
        except AttributeError:
            print("Flux metrics need true_flux and pred_flux attributes. Run the predict_fluxes function before evaluating.")

        fig, ax = plt.subplots(1,1,figsize = (6,4))
        ax.set_title("Bias across quantiles for " + self.data.site, fontsize=10)
        ax.plot(np.array(bias)*1e6, c = "#222255") 
        ax.set_ylabel("MBE, (micro mol/mol)", fontsize=9) 
        ax.set_xlabel(f"{quantiles}-Quantile", fontsize=9) 
        ax.axhline(0, c = "grey", alpha = 0.8)

        if quantiles < 6:
            ticks = np.linspace(0, 1, quantiles+1)[:-1]
            names = [f"{n}th" for n in np.arange(quantiles)+1 ]
            print(ticks)
            ax.set_xticks(ticks*quantiles, names)
        else:
            ticks = np.linspace(0, 1, quantiles+1)[:-1][::2]
            names = [f"{n}th" for n in np.arange(quantiles)+1 ][::2]
            ax.set_xticks(ticks*quantiles, names)



def predict_tree(clf, data, inputs, tree, hours_back=None, return_truths=False):
    """
    Predict output for a single trained GBRT (ie the regressor for a single cell) using parameters provided. Returns prediction. Parallel function to train_tree
    
    """

    sizediff = int((data.metsize-data.size)/2)
    centre = int(data.size/2)

    (x,y) = np.unravel_index(tree, (data.size,data.size))
    xtree, ytree = x + sizediff,  y + sizediff
    around_tree_indeces = []
    for i in [-2,0,+2]:
        for j in [-2,0,+2]:
            around_tree_indeces.append((xtree+i, ytree+j))

    # fixed indices (same for every tree)
    around_measurement_indeces = [(centre+sizediff,j+centre+sizediff) for j in [0,-1,1]] + [(j + centre+sizediff, centre+sizediff) for j in [-1,1]] + [(i+centre+sizediff, j+centre+sizediff) for (i,j) in [(-1,-1), (+1,+1), (+1,-1), (-1,+1)]]


    indeces = around_measurement_indeces + around_tree_indeces

    ## find flattened indeces from 2D coordinates for a single input (ie indices to keep in a metsize x metsize grid)
    idx_list_tree =  [np.ravel_multi_index([x,y], (data.metsize, data.metsize)) for (x,y) in indeces]

    ## repeat indeces for each of the concatenated inputs
    idxs = [i + (data.metsize**2)*n for n in range(int(np.shape(inputs)[1]/(data.metsize**2))) for i in idx_list_tree]

    ## select inputs
    inputs_here = inputs[:, idxs]

    predictions = clf.predict(inputs_here)

    predictions[predictions < 0] = 0 

    if not return_truths:
        return predictions
    if return_truths:
        assert hours_back != None, print("If returning truths, please pass hours_back parameter")
        truths = data.fp_data[hours_back:-3, tree]  
        return truths, predictions
