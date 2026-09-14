# Neural Decoding of EEG Signals

This repository contains MATLAB code developed for the Master's course "Neural Decoding of EEG Signals" at University of Fribourg. The project applies machine learning methods to EEG data, with a focus on linear Support Vector Machine (SVM) classification and time-resolved neural decoding.

The analyses are divided into three parts:

### Part 1: Binary Classification

A linear SVM is used to perform binary classification of EEG signals. Decoding performance is evaluated over time and visualized to investigate when information relevant to the classification is represented in the EEG signal.

### Part 2: Electrode Regions

The analysis is extended to compare decoding performance between **posterior and frontal electrodes**. Results from the two electrode groups are visualized and compared to investigate potential differences in the spatial distribution of neural representations.

### Part 3: Without Pseudo-Averaging

The decoding analysis is repeated **without pseudo-averaging**. Results are compared with those from Part 1 to examine how pseudo-averaging influences decoding performance and the resulting interpretation.

The repository includes the MATLAB scripts used for preprocessing, classification, decoding performance analysis, and visualization. as well as the data from the Things dataset (publicly available).

### How to run:
On MATLAB, import the data into the same folder as the scripts. Run the scripts in their order.
