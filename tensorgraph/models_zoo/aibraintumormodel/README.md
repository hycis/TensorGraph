# AI-Brain-Tumor Model 5
Brain tumor classification & segmentation using ensemble models

----

Six models trained to simultaneously produce segmentation maps & classification
on 24 distinct tumor types, grouped into 19 classes for final output in the
ensemble. This repository contains the scripts to train each model individually.

Output of the training procedure are TF checkpoints for 6 variations of a
joint semgentation-classification model which can then be ensembled in a 
separate script.

Folder content:  
1. main_train.py - Main training launcher  
2. run_mpi.sh; run_nonmpi.sh - Script to run MPI/non-MPI processes on da Vinci  
3. nn/ - Folder containing model, data, and actual training scripts  
4. model_C3/; model_C4/; model_C4R/; model_C5/; model_C5XS/; model_CR/ - Folders with training configuration INIs for each model 

To train, go to each model folder and run ../run_mpi.sh <GPUs e.g. "1,2"> ../main_train.py <model_training_config.ini>"
