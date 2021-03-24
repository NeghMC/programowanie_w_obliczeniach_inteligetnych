# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from scipy.stats import truncnorm, uniform
from csv import writer
from math import pi
from numpy import sin, cos, zeros


def surface_point(orientation="vert", num_points=2000, size=(10, 10)):
    width, height = size
    a = uniform(0, width).rvs(size=num_points)
    b = uniform(0, height).rvs(size=num_points)
    if orientation == "vert":
        return zip(a, b, zeros(num_points))
    elif orientation == "hor":
        return zip(zeros(num_points), a, b)
    else:
        return None


def cylinder_points(num_points=2000, radius=10, height=20):
    radius = uniform(0, radius).rvs(size=num_points)
    theta = uniform(0, 2 * pi).rvs(size=num_points)

    cyl_mean = 0
    cyl_std = 5
    z = truncnorm(a=(0 - cyl_mean) / cyl_std, b=(height - cyl_mean) / cyl_std, loc=cyl_mean, scale=cyl_std) \
        .rvs(size=num_points)

    return zip(radius * cos(theta), radius * sin(theta), z)


def main():
    cloud_points = surface_point(orientation="vert", num_points=5000)
    # cloud_points = cylinder_points(5000)
    with open('cloud.xyz', 'w', encoding='utf-8', newline='\n') as csvfile:
        csvwriter = writer(csvfile)
        for p in cloud_points:
            csvwriter.writerow(p)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
