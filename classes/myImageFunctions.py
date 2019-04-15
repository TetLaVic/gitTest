import numpy as np
from PIL import Image
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt


def find_local_max(fname, neighborhood_size, threshold, *args, **kwargs):

    # data = Image.open(fname).convert('LA')
    # data.save('greyscale.png')

    plotx = kwargs.get('x', None)
    ploty = kwargs.get('y', None)

    data = fname

    data_max = filters.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    # np.where(maxima)

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y = [], []
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1)/2
        y.append(y_center)

    plt.figure(0)
    if plotx is not None or ploty is not None :
        plt.pcolormesh(plotx, ploty, data)
    else:
        plt.pcolormesh(data)
    plt.savefig('rdEYd2.png', bbox_inches='tight')

    # plt.autoscale(False)
    # plt.plot(x, y, 'ro')
    # plt.savefig('rdEYd3.png', bbox_inches='tight')


if __name__ == "__main__":

    fname = 'rdEYd.png'
    neighborhood_size = 5
    threshold = 1000

    find_local_max(fname, neighborhood_size, threshold)


