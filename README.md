**Training CNNs with Pytorch** 
------------------------------
This repository provides a modular base to train CNNs using Pytorch. It has integration with [TensorboardX](https://github.com/lanpa/tensorboardX) which allows Tensorboard style visualisations while having the rest of the code remain in pythonic Pytorch.  

The list of files in the directory and their functions are described below. 

pytorch\_tensorflow.yml
=======================
Anaconda config file that can be used to setup a conda environment with all the required dependencies. The list of dependencies can be found in this file.

main
====
Contains main function that can is used to call functions in the rest of the files 

param\_parser
==================
Reads the config file present in the configs folder that holds the configuration and returns an object with all the parameters that were passed in. The config file to be used can be specified as `python3 main.py --config-file "name of config file"` when calling the function. 

config.ini
==========
> Dataset
- **Dataset** : Name of dataset that will be recognised by code
- **Dataset\_Location** : Folder which holds the location of the dataset. This will also depend on how the dataset is being read in the input\_preprocess.py file
> CNN
- **Architecture** : Name of the CNN being trained as recognised by code in the model\_creator.py file 
- Various other architecture related parameters that may or may not be relevant depending on parsing in the model\_creator.py file
> Training\_Hyperparameters
- Most training\_hyperparameters are self explanatory 
> Pytorch\_Parameters
- **Manual\_Seed** : Random number generator seed (set in main.py)
- **Data\_Loading\_Workers** : Pytorch parameter to set number of parallel threads reading in data 
- **GPU\_ID** : Which GPU to use. If multiple specify with comma separation as 0,1,... 
- **Checkpoint\_Path** : Used in conjunction with **Test\_Name** if **Resume**, **Branch** and **Evaluate** are all False. If used, it will create a new folder with path "Checkpoint\_Path/Test\_Name" where training log files will be placed 
- **Test\_Name** : Specify name of new training being run, this can be anything
- **Pretrained** : Used if any one of **Resume**, **Branch**, or **Evaluate** are set to True. This needs to be set as path to a \*-model.pth.tar file that holds the checkpoint from which training needs to continue or evaluation needs to be performed. 
- **Resume** : If set to True, training will resume from the epoch specified in the checkpoint file in **Pretrained** and the state will be taken in from the checkpointed state. The values for hyperparameters specified in this config file with be ignored. 
- **Branch** : If set to True, training will fork from the epoch specified in the checkpoint file in **Pretrained** and will continue with the hyperparameters specified in the config file. Previous state that was stored will be ignored. 
- **Evaluate** : If set to True, inference will be performed on the checkpointed model in **Pretrained**, with hyperparameters specified in this config file. 
- **Tee\_Printing** : If a csv file is specified here, it overloads the print function in Python such that if the string passed into the print function starts with a **~**, the print function will write both to stdout and to the csv specified here. If left as None, the print function is not overloaded. 
> *Note*: Only one of **Resume**, **Branch**, or **Evaluate** can be set to True at any given time. Directory structure for checkpointing will be specified in the section describing the checkpointing.py file.

model\_creator
==============
Looks at the **cnn** section of the config file and loads the model specified in the **models** folder. Within the models folder, the *\_\_init\_\_.py* in the *models/dataset* folder should have the `from .dataset.py import *` command for each model that you wish to use, and within the **dataset.py** file, the *\_\_all\_\_* value needs to be set to the name of the dataset to be imported

checkpointing
=============
Defines a class that is instantiated in the **main** which holds the state during training as well as deals with checkpointing. 
Whenever a new test is created, i.e. **Branch**, **Resume** and **Evaluate** are all False, the directory is set to **Checkpoint\_Path/Test\_Name/orig**. In here, after every epoch, two files are stored with the names *(epoch\_number)-model.pth.tar* and *(epoch\_number)-state.pth.tar*. 

If a **Resume** is called, then whichever directory the *model.pth.tar* file was in, the new checkpoints are placed in that directory itself. The code checks to ensure that the checkpoint file passed to resume from is the last epoch that is stored in that directory. If you wish to resume from a different epoch, a **Branch** command needs to be used. 

If a **Branch** is called, then at the same level as the **orig** directory, a new directory is created with name *(start_epoch_number)-(version)*. So if multiple different branches are created with the same start epoch, the version number is incremented by 1 each time. The checkpoint of the start epoch is copied from the **orig** directory into this new directory, and training is resumed from the following epoch. The relevant files in the old log file are also copied over, so the new logfile within this new directory is complete with history data and new data. 









