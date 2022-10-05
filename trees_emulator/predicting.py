import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator


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



def plot_flux(MakePredictions, LoadData, month):
    """
    Plot mole fraction from LPDM footprints and emulated footprints for a particular month.

    Takes as inputs a MakePredictions object that has attributes true_flux and pred_flux (ie predict_fluxes has been run),
    a LoadData object and the month to plot, as an integer (January is 1, February 2 etc).

    """
    fig, axis = plt.subplots(1,1,figsize = (15,4))

    fontsizes = {"title":17, "labels": 15, "axis":10}


    month = np.argwhere(pd.DatetimeIndex(LoadData.met.time.values[MakePredictions.jump:-3]).month == month)
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





