import numpy as np
import pptk
import pylas
import pandas as pd
import matplotlib.colors as mcolors

# Tutorial: https://towardsdatascience.com/guide-to-real-time-visualisation-of-massive-3d-point-clouds-in-python-ea6f00241ee0


def prepare_data(input_path):
    point_cloud = pylas.read(input_path)
    points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
    #print("\nPoints:", points)
    colors = np.vstack((point_cloud.red, point_cloud.green, point_cloud.blue)).transpose()
    #print("\nColors:", colors)
    #normals = np.vstack((point_cloud.normalx, point_cloud.normaly, point_cloud.normalz)).transpose()
    return point_cloud, points, colors


def select_points(point_cloud, classification):
    df = pd.DataFrame(point_cloud.points)
    df = df[['X', 'Y', 'Z', 'raw_classification', 'red', 'green', 'blue']]
    df = df.loc[df['raw_classification'] == classification]

    df_points = df[['X', 'Y', 'Z']]
    df_colors = df[['red', 'green', 'blue']]

    class_points = df_points.to_numpy()
    class_colors = df_colors.to_numpy()

    return class_points, class_colors


def pptkviz(points, colors):
    v = pptk.viewer(points)
    v.attributes(colors/65535)
    v.set(point_size=0.01, bg_color=[0, 0, 0, 0], show_axis=0, show_grid=0)
    return v


def computePCFeatures(points, colors, knn=10, radius=np.inf):
    normals = pptk.estimate_normals(points, knn, radius)
    idx_ground = np.where(points[..., 2] > np.min(points[..., 2] + 0.3))
    idx_normals = np.where(abs(normals[..., 2]) < 0.9)
    idx_wrongly_filtered = np.setdiff1d(idx_ground, idx_normals)
    # filtering out the ground points
    common_filtering = np.setdiff1d(idx_normals, idx_wrongly_filtered)
    return points[common_filtering], colors[common_filtering]


# color is encoded in 16 bits (instead of 8 bits) -> 65536 instead of 256
# color map with 19 different colors
color_map = [
    [65536, 0, 0],          # red
    [0, 65536, 0],          # green
    [0, 0, 65536],          # blue
    [65536, 65536, 0],      # yellow
    [65536, 0, 65536],      # magenta
    [0, 65536, 65536],      # cyan
    [32768, 32768, 0],      # olive
    [32768, 0, 32768],      # purple
    [0, 32768, 32768],      # teal
    [43690, 43690, 0],
    [43690, 0, 43690],
    [0, 43690, 43690],
    [65536, 43690, 0],
    [65536, 0, 43690],
    [43690, 65536, 0],
    [43690, 0, 65536],
    [0, 65536, 43690],
    [0, 43690, 65536],
    [65536, 65536, 65536]   # white
]


# function to return a specific color based on the Label
def colorise(row):
    return color_map[row['Label']]


