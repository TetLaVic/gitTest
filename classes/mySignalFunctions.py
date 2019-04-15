import numpy
import matplotlib.pyplot as plt


# Function that returns the position of a zero crossing (maybe add +1/m for more precise position)
def get_zero_crossings(array):

    zero_crossings = numpy.where(numpy.diff(numpy.sign(array)))[0]

    x = numpy.arange(len(array)-1)
    m = numpy.diff(array)
    t = array[:-1] - m * x
    zero_crossings_precise = -t/m

    zero_crossings_precise = zero_crossings_precise[zero_crossings]
    freqs = numpy.diff(zero_crossings)

    return zero_crossings_precise, freqs


if __name__ == "__main__":

    a = [1, 2, 1, 1, -3, -4, 7, 8, 9, 10, -2, 1, -3, 5, 6, 7, -10, 2]

    data = pd.read_csv(file_path, header=None, sep="\s+")

    plt.figure()
    plt.plot(a)
    plt.show()

    z_c, freqs = get_zero_crossings(a)

    print('\n')
    print('Amount: ', len(z_c))
    print('Locations: ', z_c)
    print('Distances between zero crossings: ', freqs)


