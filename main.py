# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from scipy.stats import truncnorm, uniform
from csv import writer
from math import pi
from numpy import sin, cos, full, array, cross, dot, sqrt
from numpy.random import randint
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


# proba samodzielnej implementacji ransaca
# https://www.youtube.com/watch?v=9D5rrtCC_E0
# https://en.wikipedia.org/wiki/Random_sample_consensus
def ransacForPlane(data, max_iter=100):
    # returns vector of a plane passing through 3 given points
    def planeVector(points):
        p = array(points[0]) - array(points[1])
        q = array(points[0]) - array(points[2])
        v = cross(p, q)
        D = dot(array(points[0]), v)
        ret = v.tolist()
        ret.append(D)
        return ret

    def distanceFromPlane(plane_vector, point):
        return abs(dot(array(plane_vector[0:3]), array(point))) / sqrt(
            plane_vector[0] ** 2 + plane_vector[1] ** 2 + plane_vector[2] ** 2)

    bestPlane = None
    bestScore = 0

    for iter in range(0, max_iter):
        drawnIndices = randint(len(data), size=3)
        maybeInliers = [data[i] for i in drawnIndices.tolist()]
        selectedPlane = planeVector(maybeInliers)
        alsoInliers = []  # empty set
        for point in data:
            if distanceFromPlane(selectedPlane, point) < 5:
                alsoInliers.append(point)

        modelScore = len(alsoInliers)
        if modelScore > bestScore:
            bestScore = modelScore
            bestPlane = selectedPlane

    return bestPlane


def main():
    # generating cloud points
    cloud_points = []
    cloud_points.extend(surface_point(offset=(10, 20, 10), orientation="vert", num_points=5000))
    cloud_points.extend(surface_point(offset=(-10, -20, -10), orientation="hor", num_points=5000))
    cloud_points.extend(cylinder_points(num_points=5000))
    with open('cloud.xyz', 'w', encoding='utf-8', newline='\n') as csvfile:
        csvwriter = writer(csvfile)
        for p in cloud_points:
            csvwriter.writerow(p)

    # separating clouds to clusters
    data = array(cloud_points)
    prediction = KMeans(n_clusters=3).fit_predict(array(cloud_points))
    # plt.figure()
    # plt.scatter(data[red, 0], data[red, 1], c="r")
    # plt.scatter(data[blue, 0], data[blue, 1], c="b")
    # plt.scatter(data[green, 0], data[green, 1], c="g")
    # plt.show()
    for i in range(3):
        surf = ransacForPlane(data[prediction == i].tolist(), 300)
        if surf != None:
            print(surf[0], "*x+", surf[1], "*y+", surf[2], "*z=", surf[3])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
