# LPDM-emulation-trees
This repository contains the scripts required to reproduce the results presented in "A machine learning emulator for Lagrangian particle dispersion model footprints: A case study using NAME" by Elena Fillola, Raul Santos-Rodriguez, Alistair Manning, Simon O'Doherty and Matt Rigby.


## Requirements and Creating the Conda Environment
This code requires Python 3 to run. The easiest way to get the right packages installed is to use Anaconda. To get the repository and environment ready, follow these steps:

1. Clone the repository, using: git clone https://github.com/elenafillo/LPDM-emulation-trees.git
2. Move into the repository and create the Conda environment: conda env create -f environment.yml
3. Activate the environment: conda activate trees_emulator

## Data
You can download the sample data accompanying this code from https://doi.org/10.5281/zenodo.7254330 . To use with the code, just download and unzip the data.zip in the repository folder. This file follows the default folder structure:

        /data
            /fps
               site-siteheight*yearmonth.nc (eg MHD-10magl*201603.nc)
           /met
              domain_Met_metheight_yearmonth.nc (eg EUROPE_Met_10magl_201603.nc)
              domain_verticalgradients_yearmonth.nc (eg EUROPE_verticalgradients_201603.nc)
           /fluxes
              regridded_fluxes.nc (no naming default)
           /trained_models
              site.txt (eg MHD.txt, this is the default saving name)
           /emulated_fps
              site_yearmonth.nc (eg MHD_201603.nc, this is the default saving name)    
           
 If you want to use folders and files with different structure or naming, you can pas the paths to LoadData (see below).             
              

## Scripts

### Training
A new emulator can be trained using the train_emulator.py script. You can train the emulator using the default parameters and default data folders, and specifying only the site code (e.g. Mace Head is MHD) and year(s) to train on (e.g. 2014 if a single year or 201[4-5] if multiple).

        python train_emulator.py MHD 201[4-5]
        
If you want to use specific folders or parameters, you can specify them, for example:

        python train_emulator.py MHD 201[4-5] --freq 3 --size 16 --met_datadir "/path/to/folder/met_filename_format*" --fp_datadir "/path/to/folder/fp_filename_format*" --extramet_datadir "/path/to/folder/gradients_filename_format*" --save_dir "path/to/save/folder/trained_emulator.txt"
        
where freq is the sampling frequency of the inputs (eg --freq 3 means one in every three footprints is used for training, default is 2) and size is the size the domain will be cut to (default is 10), and met_datadir, fp_datadir and extramet_datadir are the folders where the three types of data are found. If no datadir arguments are passed, the default folder and name structure above is used. If you pass your own paths, note that the year and months to load are added within the code, so pass only the general naming format. save_dir is the folder and name under which you would like to save your trained model, it should end with ".txt". Note that loading training data for multiple years can use a lot of memory.

### Predicting and Evaluating
The process to predict new and evaluate footprints is outlined in the guide.ipynb document. First, the data and trained models should be loaded and the variables and fluxes set up. Predictions can be made using the MakePredictions object, which can also be used to plot the predicted footprints and fluxes and to evaluate the predictions using the paper metrics. A gif of the predictions can be produced, like the one below:
 ![prediction examples](footprints_00-07-04-2016_00-17-04-2016.gif)
 
 ### Custom training/predicting functions
The scripts here implement the algorithm described in the paper, allowing some flexibility (eg different sampling frequency or size than described). Users can create new training/predicting functions to for example use different sets of inputs or different regressors. To create a new training function, write a function in trees_emulator/training.py imitating the default function (train_tree) and replace the default function call with the custom function in train_emulator. Similarly, to predict, create a function imitating the default function (predict_tree) and replace the default function call with the custom function in the object MakePredictions.
