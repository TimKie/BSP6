import pylas
import numpy as np


def decompress(input_laz_path, output_las_path):
    laz = pylas.read(input_laz_path)
    las = pylas.convert(laz)
    las.write(output_las_path)

    return output_las_path


def describe(input_file):
    las = pylas.read(input_file)

    print("Unique classification:\t\t", np.unique(las.classification))
    print("Number of classifications:\t", len(las.classification))  # total number of classifications = number of points -> each point is classified
    print("Point Count:\t\t\t", las.header.point_count)
    print("Point Format:\t\t\t", las.point_format.id)  # https://pylas.readthedocs.io/en/latest/intro.html#point-format-3

    dim = las.point_format.dimension_names
    print("Dimension Names:\t\t", dim)

    print("\nValues of each dimension:")
    for i in range(len(dim)):
        dim_name = dim[i]
        dim_value = las[dim_name]
        print(f"{dim_name : <20}{':' : ^5}{str(dim_value) : <100}")
