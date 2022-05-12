import pandas as pd
import numpy as np
import pylas
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN, KMeans, OPTICS, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from yellowbrick.cluster import KElbowVisualizer


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

    #print("Scaled Numpy Array:\n", df_scaled)

    # Normalizing the data so that the data approximately follows a Gaussian distribution
    df_normalized = normalize(df_scaled)

    # Converting the numpy array into a pandas DataFrame
    df_normalized = pd.DataFrame(df_normalized)
    df_normalized.columns = ['X', 'Y', 'Z']

    print("Normalized dataframe:\n", df_normalized.head())

    return df_scaled, df_normalized


def elbow_method(df, model, k):
    # ask the user if he wants to use the elbow method
    el = input("\nDo you want to use the elbow method to help you choosing the appropriate value for k? (y / n): ")

    elbow = False
    valid_answer = False

    # continue to ask for a valid answer until the user gives a valid input
    while not valid_answer:
        if el.lower() == 'y' or el.lower() == 'yes':
            elbow = True
            valid_answer = True
        elif el.lower() == 'n' or el.lower() == 'no':
            elbow = False
            valid_answer = True
        else:
            el = input("Please enter a valid answer (y / n): ")

    if elbow:
        print("\nComputing the optimal value for k...")
        visualizer = KElbowVisualizer(model, k=k)
        # fit the data to the visualizer
        visualizer.fit(pd.DataFrame(df))

        # ask the user if he wants to visualize the graph of the elbow method
        show = input("\nDo you want to visualize the graph that is created by the elbow method? (y / n): ")

        show_graph = False
        valid_answer = False

        # continue to ask for a valid answer until the user gives a valid input
        while not valid_answer:
            if show.lower() == 'y' or show.lower() == 'yes':
                show_graph = True
                valid_answer = True
            elif show.lower() == 'n' or show.lower() == 'no':
                show_graph = False
                valid_answer = True
            else:
                show = input("Please enter a valid answer (y / n): ")

        if show_graph:
            print("\nTo continue with using the application, you have to close the window that shows the graph.")
            # render the graph
            visualizer.show()

        return visualizer.elbow_value_


def dbscan(df, eps, min_samples, algorithm):
    print("\n------------------------------------ DBSCAN ------------------------------------")
    print("Processing...")

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, algorithm=algorithm)
    dbscan.fit(pd.DataFrame(df))

    return dbscan.labels_


def kmeans(df, number_of_clusters, n_init, max_iter, algorithm):
    print("\n------------------------------------ KMeans ------------------------------------")
    print("Processing...")

    kmeans = KMeans(n_clusters=number_of_clusters, n_init=n_init, max_iter=max_iter, algorithm=algorithm)
    kmeans.fit(pd.DataFrame(df))

    return kmeans.labels_


def optics(df, min_samples, max_eps, cluster_method):
    print("\n------------------------------------ OPTICS ------------------------------------")
    print("Processing...")

    optics = OPTICS(min_samples=min_samples, max_eps=max_eps, cluster_method=cluster_method)
    optics.fit(pd.DataFrame(df))

    return optics.labels_


def agglomerative_clustering(df, number_of_clusters, affinity, linkage):
    print("\n--------------------------- Agglomerative Clustering ---------------------------")
    print("Processing...")

    agglomerative_clustering = AgglomerativeClustering(n_clusters=number_of_clusters, affinity=affinity, linkage=linkage)
    agglomerative_clustering.fit(pd.DataFrame(df))

    return agglomerative_clustering.labels_


def gaussian_mixture(df, n_components, covariance_type, max_iter, n_init):
    print("\n------------------------------- Gaussian Mixture -------------------------------")
    print("Processing...")

    gaussian_mixture = GaussianMixture(n_components=n_components, covariance_type=covariance_type, max_iter=max_iter, n_init=n_init)
    gaussian_mixture.fit(pd.DataFrame(df))

    return gaussian_mixture.predict(pd.DataFrame(df))
