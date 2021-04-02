# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from scipy.stats import truncnorm, uniform
from csv import writer
from math import pi
from numpy import sin, cos, full, array, cross, dot, sqrt, ndarray, empty, seterr
from numpy.random import default_rng
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def surface_point(offset=(0, 0, 0), orientation="vert", num_points=2000, size=(10, 10)):
    width, height = size
    a = uniform(0, width).rvs(size=num_points)
    b = uniform(0, height).rvs(size=num_points)
    if orientation == "vert":
        return zip(a + full(num_points, offset[0]), b + full(num_points, offset[1]), full(num_points, offset[2]))
    elif orientation == "hor":
        return zip(full(num_points, offset[0]), a + full(num_points, offset[1]), b + full(num_points, offset[2]))
    else:
        return None


def cylinder_points(offset=(0, 0, 0), num_points=2000, radius=10, height=20):
    radius = uniform(0, radius).rvs(size=num_points)
    theta = uniform(0, 2 * pi).rvs(size=num_points)

    cyl_mean = 0
    cyl_std = 5
    z = truncnorm(a=(0 - cyl_mean) / cyl_std, b=(height - cyl_mean) / cyl_std, loc=cyl_mean, scale=cyl_std) \
        .rvs(size=num_points)

    return zip(radius * cos(theta) + full(num_points, offset[0]), radius * sin(theta) + full(num_points, offset[1]),
               z + full(num_points, offset[2]))


# Own implementation of RANSAC algorithm
# https://www.youtube.com/watch?v=9D5rrtCC_E0
# https://en.wikipedia.org/wiki/Random_sample_consensus
def my_ransac(data, max_iter=100):
    # returns vector of a plane passing through 3 given points
    def get_plane_vector(points):
        p = points[1] - points[0]
        q = points[2] - points[0]
        # math
        ABC = cross(p, q)
        D = -dot(points[0], ABC)
        return array([*ABC, D])

    # returns the shortest distance from a point to a plane
    def get_distance_from_plane(plane_vector, point):
        ABC = plane_vector[0:3]
        D = plane_vector[3]
        # and more math
        distance = abs(dot(ABC, point) + D) / sqrt(dot(ABC, ABC))
        return distance

    best_plane = None
    best_score = 0

    # changing to numpy type if needed
    if type(data) is not ndarray:
        data = array(data)

    for iter in range(0, max_iter):
        # selecting 3 different random points to create a plane
        selected_plane = get_plane_vector([data[i] for i in default_rng().choice(len(data), size=3, replace=False)])
        # counting number of points being close to the surface
        model_score = sum([True for point in data if get_distance_from_plane(selected_plane, point) < 3])
        # checking how it did
        if model_score > best_score:
            best_score = model_score
            best_plane = selected_plane

    # back to list
    if best_plane is not None:
        best_plane = best_plane.tolist()

    return best_plane


def main():
    # generating cloud points
    cloud_points = [
        *surface_point(offset=(10, 20, 10), orientation="vert", num_points=5000),
        *surface_point(offset=(-10, -20, -10), orientation="hor", num_points=5000),
        *cylinder_points(num_points=5000)
    ]
    with open('cloud.xyz', 'w', encoding='utf-8', newline='\n') as csvfile:
        csvwriter = writer(csvfile)
        for p in cloud_points:
            csvwriter.writerow(p)

    # separating clouds to clusters
    data = array(cloud_points)
    prediction = KMeans(n_clusters=3).fit_predict(data)

    red_shape = data[prediction == 0]
    green_shape = data[prediction == 1]
    blue_shape = data[prediction == 2]

    # displaying them in 3D
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.set_xlim([-20, 30])
    ax.set_ylim([-30, 40])
    ax.set_zlim([-20, 30])
    ax.set_title("The cloud")

    ax.scatter(*list(zip(*red_shape)), c='r')
    ax.scatter(*list(zip(*green_shape)), c='g')
    ax.scatter(*list(zip(*blue_shape)), c='b')

    # finding best fitting planes and displaying vectors of the planes
    ax = fig.add_subplot(122, projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_title("Normalized vectors of the planes")

    surf = my_ransac(red_shape, 300)
    ax.quiver(0, 0, 0, *surf[0:3], pivot='tail', color='r', normalize=True)
    surf = my_ransac(green_shape, 300)
    ax.quiver(0, 0, 0, *surf[0:3], pivot='tail', color='g', normalize=True)
    surf = my_ransac(blue_shape, 300)
    ax.quiver(0, 0, 0, *surf[0:3], pivot='tail', color='b', normalize=True)

    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
