# Project Road Segmentation

The goal of this project is to perform road
segmentation on satellite images from google maps.

This directory contains final report and all code necessary to rerun our experiments and get the final model that we submitted on [AIcrowd challenge](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation).

## Contributors

* Florian Grötschla
* Adrien Bertaud
* Maximilian Wessendorf

## Report

* **Report.pdf**: final report.

## Usage

**main.ipynb** allows to train/test the experiments and the final model, generate the submission file and several images used in the report.

The code itself is split into several directories:
* **data/**: Contains the training and test images and ground truth for the training images
* **playground/**: Contains notebooks that we used to get preliminary results and to play around, in addition it contains the notebook **model_eval_plots** which was used to generate all of our plots for the report. The plots for the report can be found in **playground/images/**. 
* **results/**: Contains the models and (pickled) histories for the runs of the experimental setup and the final model we trained and submitted (in the **final** subfolder), also includes the final submission
* **src/**: Contains the main code and program logic used in the project, split into several python files. These files are included in the main notebook

## Software versions

All our tests were run in google colab. We used the following software versions:

* **python**: 3.6.9 (default, Oct  8 2020, 12:12:24) [GCC 8.4.0]
* **tensorflow**: 2.3.0
* **keras**: 2.4.3
* **numpy**: 1.18.5
* **sklearn**: 0.22.2.post1