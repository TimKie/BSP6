from File_Functions import *
from Visualization import *
from Clusertering import *


# ------------------------------------------------ Basic File Processing -----------------------------------------------
input_path = input("Please enter the path of the LAZ file that you want to process: ")     # example: ../lidar2019-ndp-c30-r7-ll93500-67000-epsg2169_(with_water)/LIDAR2019_NdP_94500_68000_EPSG2169.laz

# for development (quickly test the application with inputting a path each time)
if input_path == "1":
    input_path = 'Files/LAS_Files/test_file.las'
elif input_path == "2":
    input_path = 'Files/LAS_Files/test_file2.las'
elif input_path == "3":
    input_path = 'Files/LAS_Files/test_file3.las'
elif input_path == "4":
    input_path = 'Files/LAS_Files/test_file4_with_water.las'

las_file = decompress(input_path, "Files/LAS_Files/output_file.las")

# ask if the user wants to get a description of the file
des = input("\nDo you want to get a description of the decompressed LAS file? (y / n): ")

valid_answer = False

# continue to ask for a valid answer until the user gives a valid input
while not valid_answer:
    if des.lower() == 'y' or des.lower() == 'yes':
        valid_answer = True
        print("\n--------------------------------------- DESCRIPTION OF THE LAS FILE ---------------------------------------\n ")
        describe(las_file)
    elif des.lower() == 'n' or des.lower() == 'no':
        valid_answer = True
    else:
        des = input("Please enter a valid answer (y / n): ")


classes = np.unique(pylas.read(las_file).classification)


# ---------------------------------------------------- Visualization ---------------------------------------------------
# ask if the user wants to visualize the file (add option to enter a class and only visualize the point of this class)
vis = input("\nDo you want to visualize the raw dataset? (y / n): ")

visualization = False
valid_answer = False

# continue to ask for a valid answer until the user gives a valid input
while not valid_answer:
    if vis.lower() == 'y' or vis.lower() == 'yes':
        visualization = True
        valid_answer = True
    elif vis.lower() == 'n' or vis.lower() == 'no':
        visualization = False
        valid_answer = True
    else:
        vis = input("Please enter a valid answer (y / n): ")


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

    visualize_one_class = False
    valid_answer = False

    # continue to ask for a valid answer until the user gives a valid input
    while not valid_answer:
        if vis_one_class.lower() == 'y' or vis_one_class.lower() == 'yes':
            visualize_one_class = True
            valid_answer = True
        elif vis_one_class.lower() == 'n' or vis_one_class.lower() == 'no':
            visualize_one_class = False
            valid_answer = True
        else:
            vis_one_class = input("Please enter a valid answer (y / n): ")

    # visualize only points from a specific class
    if visualize_one_class:
        print("\nPossible classes of the data points:", classes)
        class_of_points = input("Please enter the integer corresponding to the class of your choice: ")

        valid_answer = False
        # continue to ask for a valid answer until the user gives a valid input
        while not valid_answer:
            if class_of_points.isdigit():
                if int(class_of_points) in classes:
                    valid_answer = True

                    class_points, class_colors = select_points(point_cloud, classification=int(class_of_points))
                    viewer1 = pptkviz(class_points, class_colors)
                    viewer1.set(point_size=0.1)
                else:
                    class_of_points = input("Please enter the integer corresponding to the class of your choice: ")
            else:
                class_of_points = input("Please enter the integer corresponding to the class of your choice: ")
    else:
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
list_of_algorithms = ['dbscan', 'kmeans', 'optics', 'agglomerative_clustering']
print("Possible clustering algorithms:", list_of_algorithms)
algorithm = input("Please enter the name of the algorithm (of the above list) that you want to use for the clustering of the data points: ")

valid_answer = False
# continue to ask for a valid answer until the user gives a valid input
while not valid_answer:
    if algorithm.lower() in list_of_algorithms:
        valid_answer = True
    else:
        algorithm = input("Please enter a valid answer (string of the above list): ")


if algorithm.lower() == 'dbscan':
    # ask user for parameters
    eps = 0.1

    labels = dbscan(df_normalized, eps)

elif algorithm.lower() == 'kmeans':
    # ask user for parameters
    number_of_clusters = 19
    random_state = 0

    labels = kmeans(df_normalized, number_of_clusters, random_state)

elif algorithm.lower() == 'optics':
    # ask user for parameters
    min_samples = 2

    labels = optics(df_normalized, min_samples)

elif algorithm.lower() == 'agglomerative_clustering':
    # ask user for parameters
    number_of_clusters = 4
    linkage = 'single'

    labels = agglomerative_clustering(df_normalized, number_of_clusters, linkage)

print("\nNumber of cluster found:", len(np.unique(labels)))
print("Different labels of clusters:", np.unique(labels))

df['Label'] = labels.tolist()
print("\nOriginal dataframe with label:\n", df.head())


# --------------------------------------------- Visualization of Clusters ----------------------------------------------
print("\n-------------------------------------------- Cluster VISUALIZATION --------------------------------------------")

# ask user if he wants to visualize the clusters
cluster_vis = input("Do you want to visualize the computed clusters? (y / n): ")

cluster_visualization = False
valid_answer = False

# continue to ask for a valid answer until the user gives a valid input
while not valid_answer:
    if cluster_vis.lower() == 'y' or cluster_vis.lower() == 'yes':
        cluster_visualization = True
        valid_answer = True
    elif cluster_vis.lower() == 'n' or cluster_vis.lower() == 'no':
        cluster_visualization = False
        valid_answer = True
    else:
        cluster_vis = input("Please enter a valid answer (y / n): ")

if cluster_visualization:
    print("\nNavigate through the point cloud by holding and dragging the LEFT MOUSE BUTTON (rotate the viewpoint around a turntable).")
    print("Move the viewpoint by holding SHIFT and perform LEFT MOUSE BUTTON drag.")
    print("Select a region of the point cloud by holding CTRL (COMMAND on Mac) and LEFT MOUSE BUTTON while dragging a box.")
    print("\nWhen a selection was performed, press ENTER to open a new viewer with the selection and without the ground points.")
    print("Press ENTER to quit the viewer.")

    points = df[['X', 'Y', 'Z']].to_numpy()
    labels = df['Label'].to_numpy()

    # apply the function and create a new column to store the colors
    df['Color'] = df.apply(lambda row: colorise(row), axis=1)
    print("\nDataframe with color:\n", df.head())

    colors = []
    for i in range(len(df.index.values)):
        colors.append(df['Color'].iloc[i])

    colors = np.array(colors)

    viewer1 = pptkviz(points, np.array(colors))
    #viewer1.set(point_size=0.001)

    viewer1.wait()
    viewer1.close()
