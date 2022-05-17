from File_Functions import *
from Visualization import *
from Clusertering import *
import os
import tkinter as tk
from tkinter import filedialog


# ------------------------------------------------ Basic File Processing -----------------------------------------------
#input_path = input("Please enter the path of the LAZ file that you want to process or the path of the CSV file you want to visualize: ")     # example: ../lidar2019-ndp-c30-r7-ll93500-67000-epsg2169_(with_water)/LIDAR2019_NdP_94500_68000_EPSG2169.laz

print("Please select the LAZ file that you want to process or the CSV file you want to visualize.")

root = tk.Tk()
root.withdraw()
input_path = filedialog.askopenfilename()

file_type = ''

if input_path.endswith('.laz', -4) or input_path.endswith('.las', -4):
    file_type = 'LAZ'
elif input_path.endswith('.csv', -4):
    file_type = 'CSV'

# Processing of LAZ file type
if file_type == 'LAZ':
    print("\nDecompressing LAZ file...")
    las_file = decompress(input_path, "./output_file.las")

    # ask if the user wants to get a description of the file
    des = input("\nDo you want to get a description of the decompressed LAS file? (y / n): ")

    # continue to ask for a valid answer until the user gives a valid input
    description = ask_for_valid_answer(des)
    if description:
        print("\n--------------------------------------- DESCRIPTION OF THE LAS FILE ---------------------------------------\n ")
        describe(las_file)

    classes = np.unique(pylas.read(las_file).classification)


# ---------------------------------------------------- Visualization ---------------------------------------------------
    # ask if the user wants to visualize the file (add option to enter a class and only visualize the point of this class)
    vis = input("\nDo you want to visualize the raw dataset? (y / n): ")

    # continue to ask for a valid answer until the user gives a valid input
    visualization = ask_for_valid_answer(vis)

    # visualization of raw data set
    if visualization:
        print("\n------------------------------------------------ VISUALIZATION ------------------------------------------------")
        print("Navigate through the point cloud by holding and dragging the LEFT MOUSE BUTTON (rotate the viewpoint around a turntable).")
        print("Move the viewpoint by holding SHIFT and perform LEFT MOUSE BUTTON drag.")
        print("Select a region of the point cloud by holding CTRL (COMMAND on Mac) and LEFT MOUSE BUTTON while dragging a box.")
        print("\nWhen a selection was performed, press ENTER to open a new viewer with the selection and without the ground points.")
        print("Press ENTER to quit the viewer.")

        # prepare the data for the visualization (pptk library)
        point_cloud, points, colors = prepare_data(las_file)

        # ask user if he wants to visualize the whole dataset or only one class
        vis_one_class = input("\nDo you want to visualize a specific class (y / n): ")

        # continue to ask for a valid answer until the user gives a valid input
        visualize_one_class = ask_for_valid_answer(vis_one_class)

        # visualize only points from a specific class
        if visualize_one_class:
            print("\nPossible classes of the data points:", classes)
            print("""Meaning of the integer values:
    [0]	    -	unclassified points
    [2] 	-	ground points
    [3] 	-	low vegetation
    [4] 	-	medium vegetation
    [5] 	-	high vegetation
    [6] 	-	buildings
    [7] 	-	low points (noise)
    [9] 	-	water
    [13] 	-	bridges, footbridges, viaducts
    [15]	-	high voltage lines
            """)
            class_of_points = input("Please enter the integer corresponding to the class of your choice: ")

            valid_answer = False
            # continue to ask for a valid answer until the user gives a valid input
            while not valid_answer:
                if class_of_points.isdigit():
                    if int(class_of_points) in classes:
                        valid_answer = True

                        print("\nStarting visualization...")
                        class_points, class_colors = select_points(point_cloud, classification=int(class_of_points))
                        viewer1 = pptkviz(class_points, class_colors)
                        viewer1.set(point_size=0.1)
                    else:
                        class_of_points = input("Please enter the integer corresponding to the class of your choice: ")
                else:
                    class_of_points = input("Please enter the integer corresponding to the class of your choice: ")
        else:
            print("\nStarting visualization...")
            viewer1 = pptkviz(points, colors)

        # use either the viewer for one specific class or the viewer for all data points (selection possible with both)
        viewer1.wait()
        selection = viewer1.get('selected')
        print("Selection:", selection)
        print("Number of points loaded:", viewer1.get('num_points'))
        print("Camera distance to look-at point:", viewer1.get('r'))

        viewer1.close()

        if len(selection) != 0:
            print("\nFiltering...")

            selected_points, selected_colors = computePCFeatures(points[selection], colors[selection])

            print("\nSize of selection:", len(selection))
            print("Size of filtered selection:", len(selected_points))

            viewer1 = pptk.viewer(selected_points, selected_colors/65535)
            viewer1.set(point_size=0.1)
            viewer1.wait()
            viewer1.close()


# ----------------------------------------------------- Clustering -----------------------------------------------------
    print("\n------------------------------------------------- CLUSTERING --------------------------------------------------")

    print("Possible classes of the data points:", classes)
    print("""Meaning of the integer values:
    [0]	    -	unclassified points
    [2] 	-	ground points
    [3] 	-	low vegetation
    [4] 	-	medium vegetation
    [5] 	-	high vegetation
    [6] 	-	buildings
    [7] 	-	low points (noise)
    [9] 	-	water
    [13] 	-	bridges, footbridges, viaducts
    [15]	-	high voltage lines
    """)
    class_of_points = input("Please enter the integer corresponding to the class of your choice: ")

    valid_answer = False
    # continue to ask for a valid answer until the user gives a valid input
    while not valid_answer:
        if class_of_points.isdigit():
            if int(class_of_points) in classes:
                valid_answer = True
            else:
                class_of_points = input("Please enter a valid answer (integer of the above list): ")
        else:
            class_of_points = input("Please enter a valid answer (integer of the above list): ")

    # get a dataframe with only the points from the specified class
    df = get_df_of_class(las_file, classification=int(class_of_points), csv=False)

    df_scaled, df_normalized = preprocessing(df)

    print("\n----------------------------- Clustering Algorithm -----------------------------")
    dict_of_algorithms = {'dbscan': 1, 'kmeans': 2, 'optics': 3, 'agglomerative_clustering': 4, 'gaussian_mixture': 5}
    print("Possible clustering algorithms:", dict_of_algorithms)
    algorithm = input("Please enter the name or associated integer of the algorithm (of the above list) that you want to use for the clustering of the data points: ")

    valid_answer = False
    # continue to ask for a valid answer until the user gives a valid input
    while not valid_answer:
        if (algorithm.lower() in dict_of_algorithms) or (int(algorithm) in dict_of_algorithms.values()):
            valid_answer = True
        else:
            algorithm = input("Please enter a valid answer (string of the above list): ")

    if algorithm.lower() == 'dbscan' or int(algorithm) == 1:
        # ask user for parameters
        print("\nPlease enter the values for the following parameters. Press ENTER for each parameter to use the default values.")
        eps_input = input("EPS (distance between two samples): ")
        min_samples_input = input("Minimum number of samples: ")
        algorithm_input = input("Algorithm (possible values: 'auto', 'ball_tree', 'kd_tree', 'brute'): ")

        # use default parameters when the user does not provide an input
        eps = eps_input if eps_input.isdigit() else 0.1
        min_samples = min_samples_input if min_samples_input.isdigit() else 5
        algo = algorithm_input if algorithm_input in ['auto', 'ball_tree', 'kd_tree', 'brute'] else 'auto'

        labels = dbscan(df_normalized, float(eps), int(min_samples), str(algo))

    elif algorithm.lower() == 'kmeans' or int(algorithm) == 2:
        # elbow method
        elbow_value = elbow_method(model=KMeans(), df=df_normalized, k=(1, 15))
        print("\nBest value for k according to the elbow method:", elbow_value)

        # ask user for parameters
        print("\nPlease enter the values for the following parameters. Press ENTER for each parameter to use the default values.")
        number_of_clusters_input = input("Number of clusters: ")
        n_init_input = input("Number of new initialization of cluster centroids: ")
        max_iter_input = input("Maximum number of iterations: ")
        algorithm_input = input("Algorithm (possible values: 'auto', 'full', 'elkan'): ")

        # use default parameters when the user does not provide an input
        number_of_clusters = number_of_clusters_input if number_of_clusters_input.isdigit() else 4
        n_init = n_init_input if n_init_input.isdigit() else 10
        max_iter = max_iter_input if max_iter_input.isdigit() else 300
        algo = algorithm_input if algorithm_input in ['auto', 'full', 'elkan'] else 'auto'

        labels = kmeans(df_normalized, int(number_of_clusters), int(n_init), int(max_iter), str(algo))

    elif algorithm.lower() == 'optics' or int(algorithm) == 3:
        # ask user for parameters
        print("\nPlease enter the values for the following parameters. Press ENTER for each parameter to use the default values.")
        min_samples_input = input("Minimum number of samples: ")
        max_eps_input = input("Maximum EPS (distance between two samples): ")
        cluster_method_input = input("Clustering method (possible values: 'xi', 'dbscan'): ")

        # use default parameters when the user does not provide an input
        min_samples = min_samples_input if min_samples_input.isdigit() else 5
        max_eps = max_eps_input if max_eps_input.isdigit() else np.inf
        cluster_method = cluster_method_input if cluster_method_input in ['xi', 'dbscan'] else 'xi'

        labels = optics(df_normalized, int(min_samples), float(max_eps), str(cluster_method))

    elif algorithm.lower() == 'agglomerative_clustering' or int(algorithm) == 4:
        # elbow method
        elbow_value = elbow_method(model=AgglomerativeClustering(), df=df_normalized, k=(1, 15))
        print("\nBest value for k according to the elbow method:", elbow_value)

        # ask user for parameters
        print("\nPlease enter the values for the following parameters. Press ENTER for each parameter to use the default values.")
        number_of_clusters_input = input("Number of clusters: ")
        linkage_input = input("Linkage criterion (possible values: 'ward', 'complete', 'average', 'single'): ")
        # if linkage is “ward”, only “euclidean” is accepted as affinity
        affinity = 'euclidean'
        if linkage_input != 'ward' and linkage_input != '':
            affinity_input = input("Affinity (possible values: 'euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed'): ")
            affinity = affinity_input if affinity_input in ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed'] else 'euclidean'

        # use default parameters when the user does not provide an input
        number_of_clusters = number_of_clusters_input if number_of_clusters_input.isdigit() else 4
        linkage = linkage_input if linkage_input in ['ward', 'complete', 'average', 'single'] else 'ward'

        labels = agglomerative_clustering(df_normalized, int(number_of_clusters), str(affinity), str(linkage))

    elif algorithm.lower() == 'gaussian_mixture' or int(algorithm) == 5:
        # ask user for parameters
        print("\nPlease enter the values for the following parameters. Press ENTER for each parameter to use the default values.")
        n_components_input = input("Number of mixture components: ")
        covariance_type_input = input("Covariance type (possible values: 'full', 'tied', 'diag', 'spherical'): ")
        max_iter_input = input("Maximum number of iterations: ")
        n_init_input = input("Number of new initialization of cluster centroids: ")

        # use default parameters when the user does not provide an input
        n_components = n_components_input if n_components_input.isdigit() else 1
        covariance_type = covariance_type_input if covariance_type_input in ['full', 'tied', 'diag', 'spherical'] else 'full'
        max_iter = max_iter_input if max_iter_input.isdigit() else 100
        n_init = n_init_input if n_init_input.isdigit() else 1

        labels = gaussian_mixture(df_normalized, int(n_components), str(covariance_type), int(max_iter), int(n_init))

    print("\nNumber of cluster found:", len(np.unique(labels)))
    print("Different labels of clusters:", np.unique(labels))

    df['Label'] = labels.tolist()
    print("\nOriginal dataframe with label:\n", df.head())

    # apply the function and create a new column to store the colors
    df['Color'] = df.apply(lambda row: colorise(row), axis=1)


# --------------------------------------------- Visualization of Clusters ----------------------------------------------
    print("\n-------------------------------------------- CLUSTER VISUALIZATION --------------------------------------------")

    # ask user if he wants to visualize the clusters
    cluster_vis = input("Do you want to visualize the computed clusters? (y / n): ")

    # continue to ask for a valid answer until the user gives a valid input
    cluster_visualization = ask_for_valid_answer(cluster_vis)

    if cluster_visualization:
        print("\nNavigate through the point cloud by holding and dragging the LEFT MOUSE BUTTON (rotate the viewpoint around a turntable).")
        print("Move the viewpoint by holding SHIFT and perform LEFT MOUSE BUTTON drag.")
        print("Select a region of the point cloud by holding CTRL (COMMAND on Mac) and LEFT MOUSE BUTTON while dragging a box.")
        print("\nWhen a selection was performed, press ENTER to open a new viewer with the selection and without the ground points.")
        print("Press ENTER to quit the viewer.")

        print("\nStarting visualization...")

        points = df[['X', 'Y', 'Z']].to_numpy()
        labels = df['Label'].to_numpy()

        print("\nDataframe with color:\n", df.head())

        colors = []
        for i in range(len(df.index.values)):
            colors.append(df['Color'].iloc[i])

        colors = np.array(colors)

        viewer1 = pptkviz(points, np.array(colors))
        #viewer1.set(point_size=0.001)

        viewer1.wait()
        viewer1.close()


# --------------------------------------------- Cluster Result Dataframe  ----------------------------------------------
    print("\n------------------------------------------- CLUSTER RESULT DATAFRAME ------------------------------------------")
    results = input("Do you want to save the clustering results as a CSV file? (y / n): ")

    # continue to ask for a valid answer until the user gives a valid input
    save_results = ask_for_valid_answer(results)

    if save_results:
        valid_path = False
        path_to_save = input("Please enter the path to the location where you want to save the CSV files containing the clustering results: ")

        # continue to ask for a valid answer until the user gives a valid input
        while not valid_path:
            if os.path.isdir(path_to_save):
                valid_path = True
                name_of_file = input("Please enter a name for the CSV file: ")
                final_path = os.path.join(path_to_save, name_of_file) + '.csv'
                df.to_csv(final_path)
                print("\nSaved to:", final_path)
            else:
                path_to_save = input("Please enter a valid path to a directory: ")


# --------------------------------------------- Visualization of CSV File ----------------------------------------------
else:
    print("\n-------------------------------------------- CLUSTER VISUALIZATION --------------------------------------------")
    print("Navigate through the point cloud by holding and dragging the LEFT MOUSE BUTTON (rotate the viewpoint around a turntable).")
    print("Move the viewpoint by holding SHIFT and perform LEFT MOUSE BUTTON drag.")
    print("Select a region of the point cloud by holding CTRL (COMMAND on Mac) and LEFT MOUSE BUTTON while dragging a box.")
    print("\nWhen a selection was performed, press ENTER to open a new viewer with the selection and without the ground points.")
    print("Press ENTER to quit the viewer.")

    print("\nStarting visualization...")

    imported_df = pd.read_csv(input_path)

    points = imported_df[['X', 'Y', 'Z']].to_numpy()

    colors = []
    # convert the 'Color' column of the imported dataframe to a numpy array of lists that contain the RGB integer values
    for i in range(len(imported_df.index.values)):
        c = imported_df['Color'].iloc[i]
        c = c.strip('][').split(', ')
        c = [int(x) for x in c]
        colors.append(c)

    colors = np.array(colors)

    viewer1 = pptkviz(points, np.array(colors))

    viewer1.wait()
    viewer1.close()


# delete the las file (output_file.las) that was created at the beginning of the script (decompression)
os.remove("./output_file.las")
