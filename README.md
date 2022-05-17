# Bachelor Semester Project 6
The goal of this project is to implement a Pyhton script that is able to process LiDAR data stored in LAZ/LAS files. The processing includes clustering of the data points and the visualization of those clusters.

## Functionalities
 - Decompression of LAZ to LAS files
 - Visualization of the whole LiDAR dataset (or of a specific class of the dataset)
 - Clustering of the data points belonging to a specific class of the datatset
 - Choosing out of different cluserting algorithms (DBSCAN, KMeans, OPTICS, Agglomerative Clustering and Gaussian Mixture)
 - Elbow method to choose optimal number of clusters (for KMeans and Agglomerative Clustering)
 - Input of custom values for the most important parameters of the chosen clustering algorithm
 - Visualization of the processed (clustered) dataset
 - Exporting of the processed dataset containing the clustering information in CSV format
 - Importing of a previosuly exported dataset (CSV file) in order to visualize it

## Installation
There is only one step that has to be performed before being able to run this application/script:
- Clone (download) this Github project

## Pre-requisites
- The Python version 3.6 has to be installed on the computer
- The following libraries have to be installed :
  - matplotlib 3.3.4
  - numpy 1.19.5
  - pandas 1.1.5
  - pptk 0.1.0
  - pylas 0.4.3
  - scikit_learn 0.24.2
  - yellowbrick 1.3.post1


## Usage
To run this application on your computer:
1. Open the terminal
2. In the terminal, go to the location where the code was cloned (downloaded) (the location of the *main.py* file)
3. (If the libraries listed above are not yet installed on your computer, execute the command ``python3.6 -m pip install -r requirements.txt`` to install the libraries for Python 3.6)
4. Execute the command ``python3.6 main.py``
