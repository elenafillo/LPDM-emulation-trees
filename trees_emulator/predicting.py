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
        - predict_fluxes: convolute footprints with fluxes to obtain above-baseline mole fraction in area
    """
    def __init__(self, clfs, data, inputs, jump):
        self.inputs = inputs
        self.clfs = clfs
        
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
        for n in range(len(clfs)):
            # find 2D coordinates from flattened tree number
            (x,y) = np.unravel_index(n, (self.size,self.size))
            xtree, ytree = x + sizediff,  y + sizediff
            around_tree_indeces = []
            for i in [-2,0,+2]:
                for j in [-2,0,+2]:
                    around_tree_indeces.append((xtree+i, ytree+j))
                
            indeces = around_measurement_indeces + around_tree_indeces

            ## find flattened indeces from 2D coordinates for a single input (ie indices to keep in a metsize x metsize grid)
            idx_list_tree =  [np.ravel_multi_index([x,y], (self.metsize, self.metsize)) for (x,y) in indeces]
            ## repeat indeces for each of the concatenated inputs
            idxs = [i + (self.metsize**2)*n for n in range(int(np.shape(self.inputs)[1]/(self.metsize**2))) for i in idx_list_tree]

            ## select inputs and predict
            inputs_here = self.inputs[:, idxs]
            self.predictions[:,n] = self.clfs[n].predict(inputs_here)
        
        self.predictions[self.predictions < 0] = 0 

    def predict_fluxes(self, flux, units_transform = "default"):
        ## convolute predicted footprints and fluxes, returns two np arrays, one with the true flux and one with the emulated flux, of shape (n_footprints,)
        ## flux is an array, regridded and cut to the same resolution and size of the footprints
        ## units_transform can be None (use fluxes directly), "default" (performs flux*1e3 / CH4molarmass) or another function (which should return an array of the same shape as the original flux)
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



def plot_flux(MakePredictions, LoadData, month, year=None):
    """
    Plot mole fraction from LPDM footprints and emulated footprints for a particular month.

    Takes as inputs a MakePredictions object that has attributes true_flux and pred_flux (ie predict_fluxes has been run),
    a LoadData object and the month and year to plot, as an integer (January is 1, February 2 etc and 2015, 2016 etc). The year is not necessary if the LoadData object contains only one year.

    """
    fig, axis = plt.subplots(1,1,figsize = (15,4))

    fontsizes = {"title":17, "labels": 15, "axis":10}

    if year==None:
        if type(LoadData.year) == int:
            year = LoadData.year
        else:
            print("No year was passed but the LoadData object comprises more than one year. Please pass a year.")

    month = np.argwhere((pd.DatetimeIndex(LoadData.met.time.values[MakePredictions.jump:-3]).month == month) & (pd.DatetimeIndex(LoadData.met.time.values[MakePredictions.jump:-3]).year == year))
    assert len(month) > 0, "It seems there is no data in the LoadData object for this month and year."
    dates = pd.DatetimeIndex(LoadData.met.time.values[MakePredictions.jump:-3][month])
    month_idxs = [month[0][0],month[-1][0]+1]
    
    try:
        axis.plot(dates, 1e6*MakePredictions.true_flux[month_idxs[0]:month_idxs[1]], label = "using LPDM-generated footprints", linewidth=2 ,c="#2c6dae")
        axis.plot(dates, 1e6*MakePredictions.pred_flux[month_idxs[0]:month_idxs[1]], label = "using emulated footprints", linewidth=2 ,c="#989933")
    except AttributeError:
        print("MakePredictions object needs true_flux and pred_flux attributes. Run the predict_fluxes function before plotting.")
    
    axis.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.WE))
    axis.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    axis.xaxis.set_minor_locator(mdates.DayLocator(bymonthday=range(1, 32)))

    axis.yaxis.offsetText.set_fontsize(0)
    
    axis.set_ylim([0,0.0000001+1e6*np.max([MakePredictions.pred_flux[month_idxs[0]:month_idxs[1]], MakePredictions.true_flux[month_idxs[0]:month_idxs[1]]])])

    axis.set_ylabel('Above baseline methane concentration, (micro mol/mol)', fontsize=fontsizes["axis"])
    axis.set_title('Above baseline methane concentration for ' + LoadData.site)

    axis.legend()

    plt.show()

def plot_footprint(MakePredictions, LoadData, date):

    if type(date) == str:
        try:
            date = datetime.strptime(date, "%H:00 %d/%m/%Y")
        except:
            print("Something was wrong with the date. Please pass it in format hour:00 day/month/year (eg 15:00 1/3/2016)")
        
        idx = pd.DatetimeIndex(LoadData.met.time.values[MakePredictions.jump:-3]).get_loc(date)

    if type(date) == int:
        idx = date
        date = pd.DatetimeIndex(LoadData.met.time.values[MakePredictions.jump:-3])[idx]

    #print(date, idx, type(date), type(idx))
    fig, (axr, axp) = plt.subplots(1,2,figsize = (15,15), subplot_kw={'projection':cartopy.crs.Mercator()})
    
    axr.pcolormesh(LoadData.fp_lons, LoadData.fp_lats, np.reshape(MakePredictions.truths[idx,:], (10,10)), transform=cartopy.crs.PlateCarree(), cmap="Reds", vmax = np.nanmax(MakePredictions.truths[idx,:]), vmin=0)
    c = axp.pcolormesh(LoadData.fp_lons, LoadData.fp_lats, np.reshape(MakePredictions.predictions[idx,:], (10,10)), transform=cartopy.crs.PlateCarree(), cmap="Reds", vmax = np.nanmax(MakePredictions.truths[idx,:]), vmin=0)



    for ax in [axr, axp]:
        ax.set_extent([LoadData.fp_lons[0]-0.1,LoadData.fp_lons[-1]+0.1, LoadData.fp_lats[0]+0.1,LoadData.fp_lats[-1]+0.1], crs=cartopy.crs.PlateCarree())
        ax.set_xticks(LoadData.fp_lons[::3], crs=cartopy.crs.PlateCarree())
    
        lon_formatter = LongitudeFormatter(number_format='.1f', degree_symbol='', dateline_direction_label=True)
        ax.xaxis.set_major_formatter(lon_formatter)  
        ax.set_yticks(LoadData.fp_lats[::3], crs=cartopy.crs.PlateCarree())
        lat_formatter = LatitudeFormatter(number_format='.1f',  degree_symbol='',)
        ax.yaxis.set_major_formatter(lat_formatter)             
        ax.tick_params(axis='both', which='major', labelsize=12)   

        ax.plot(LoadData.release_lon+0, LoadData.release_lat+0, marker='o', c="w", markeredgecolor = "k", transform=cartopy.crs.PlateCarree(), markersize=5)

        ax.coastlines(resolution='50m', color='black', linewidth=2)

        

        
    
    axr.set_title("LPDM-generated footprint - "+ LoadData.site + "\n" + date.strftime("%m/%d/%Y, %H:00"), fontsize = 17)
    axp.set_title("Emulator-generated footprint - "+ LoadData.site + "\n" + date.strftime("%m/%d/%Y, %H:00"), fontsize = 17)


    #gl = axr.gridlines(draw_labels=False)
    #gl.xlocator(mticker.FixedLocator(LoadData.fp_lons))
    #gl.xformatter = LongitudeFormatter()

    cbar = plt.colorbar(c, ax=[axr, axp], orientation="vertical", shrink = 0.25, aspect = 15, pad = 0.02)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("sensitivity, (mol/mol)/(mol/m2/s)", size = 15, loc="center", labelpad = 16) 

    fig.show()



