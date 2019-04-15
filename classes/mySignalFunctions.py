import numpy
import matplotlib.pyplot as plt


# Function that returns the position of a zero crossing (maybe add +1 for crossed position)
def get_zero_crossings(array):
    zero_crossings = numpy.where(numpy.diff(numpy.sign(array)))[0]
    freqs = numpy.diff(zero_crossings)
    return zero_crossings, freqs


if __name__ == "__main__":

    a = [1, 2, 1, 1, -3, -4, 7, 8, 9, 10, -2, 1, -3, 5, 6, 7, -10, 2]

    plt.figure()
    plt.plot(a)
    plt.show()

    z_c, freqs = get_zero_crossings(a)
    print('Locations: ', z_c)
    print('Distances: ', freqs)
    print('Amount: ', len(z_c))


