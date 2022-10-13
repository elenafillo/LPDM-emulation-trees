# LPDM-emulation-trees
[In progress] This repository contains the scripts required to reproduce the results presented in "A machine learning emulator for Lagrangian particle dispersion model footprints"


## Requirements and Creating the Conda Environment
This code requires Python 3 to run. The easiest way to get the right packages installed is to use Anaconda. To get the repository and environment ready, follow these steps:

1. Clone the repository, using: git clone https://github.com/elenafillo/LPDM-emulation-trees.git
2. Move into the repository and create the Conda environment: conda env create -f environment.yml
3. Activate the environment: conda activate trees_emulator

## Data

## Scripts

### Training
A new emulator can be trained using the train_emulator.py script. You can train the emulator using the default parameters and default data folders, and specifying only the site code (for example, Mace Head is MHD) and year(s) to train on, for example 2014 if a single year or 201[4-5] if multiple.

        python train_emulator.py MHD 201[4-5]
        
If you want to use specific folders or parameters, you can specify them, for example:

        python train_emulator.py MHD 201[4-5] --freq 3 --size 16 --met_datadir "/path/to/folder/met_file_format*.nc" --fp_datadir "/path/to/folder/fp_file_format*.nc" --extramet_datadir "/path/to/folder/gradients_file_format*.nc" --save_dir "path/to/save/folder/trained_emulator.txt"
        
where freq is the sampling frequency of the inputs (eg --freq 3 means one in every three footprints is used for training, default is 2) and size is the size the domain will be cut to (default is 10), and met_datadir, fp_datadir and extramet_datadir are the folders where the three types of data are found. save_dir is the folder and name under which you would like to save your trained model, it should end with ".txt". Note that loading training data for multiple years can use a lot of memory.

### Predicting and Evaluating
The process to predict new and evaluate footprints is outlined in the guide.ipynb document. First, the data and trained models should be loaded and the variables and fluxes set up. Predictions can be made using the MakePredictions object, which can also be used to plot the predicted footprints and fluxes and to evaluate the predictions using the paper metrics. A gif of the predictions can be produced, like the one below:
 ![prediction examples](footprints_00-07-04-2016_00-17-04-2016.gif)
 
 ### Custom training/predicting functions
 The scripts here implement the algorithm described in the paper, allowing some flexibility (eg different sampling frequency or size than described). Users can create new training/predicting functions to for example use different sets of inputs or different regressors. To create a new training function, write a function trees_emulator/training.py imitating the default function (train_tree) and replace the the default function call with the custom function in train_emulator. To predict, create either a prediction class imitating the default (MakePredictions in trees_emulator/predicting.py) or a custom function within the default class.
