import pandas as pd
import numpy as np
import pylas
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans


def get_df_of_class(input_file, classification, csv):
    print("\n--------------------------- Cluster Data Description ---------------------------")
    las = pylas.read(input_file)

    # Create a dataframe from the point values of the las file
    df = pd.DataFrame(las.points)
    if csv:
        df.to_csv('las_input_file_csv')

    # Use only columns that are relevant (X, Y, Z, Classification)
    df = df[['X', 'Y', 'Z', 'raw_classification']]

    # Use only points from one classification
    #print("\nClassification Labels:", np.unique(las.classification))
    df = df.loc[df['raw_classification'] == classification]
    print("Dataframe with relevant columns for clustering:\n", df.head())
    print("Number of data points:", len(df.index))

    return df


def preprocessing(dataframe):
    print("\n-------------------------------- Preprocessing ---------------------------------")
    df = dataframe[['X', 'Y', 'Z']]  # also tried with only X and Y coordinates for the clustering

    # Scaling the data to bring all the attributes to a comparable level
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    print("Scaled Numpy Array:\n", df_scaled)

    # Normalizing the data so that the data approximately follows a Gaussian distribution
    df_normalized = normalize(df_scaled)

    # Converting the numpy array into a pandas DataFrame
    df_normalized = pd.DataFrame(df_normalized)
    df_normalized.columns = ['X', 'Y', 'Z']

    print("\nNormalized dataframe:\n", df_normalized.head())

    return df_scaled, df_normalized


def dbscan(df, eps):
    print("\n------------------------------------ DBSCAN ------------------------------------")
    print("Processing...")

    dbscan = DBSCAN(eps=eps)
    dbscan.fit(pd.DataFrame(df))

    return dbscan.labels_


def kmeans(df, number_of_clusters, random_state):
    print("\n------------------------------------ KMeans ------------------------------------")

    print("Processing...")

    kmeans = KMeans(n_clusters=number_of_clusters, random_state=random_state)
    kmeans.fit(pd.DataFrame(df))

    return kmeans.labels_
