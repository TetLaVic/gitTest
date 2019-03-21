# %% import the libraries
from classes.format import bcolors
import matplotlib.pyplot as plot
import pandas as pd
from scipy import signal
import os
import numpy as np
import math
import tkinter as tk
from tkinter import filedialog

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar

# %% Parameters
freq_tx = 77000  # in Hz
samplingFrequency = 1000000 / 10  # Sampling Frequency

# %% Constants
c = 3 * 10 ** 8

# %% Open file select window
root = tk.Tk()
root.withdraw()
file_paths = filedialog.askopenfilenames(parent=root, title='Choose datasets')

for file_path in file_paths:

    # Read data file with pandas
    data = pd.read_csv(file_path, header=None, sep="\s+")

    downsampled = data.iloc[::10, :]

    data = downsampled

    save_name = os.path.basename(file_path)

    # Print preview
    printout = data.head(n=5)
    print("\n" + bcolors.OKGREEN + "Preview:\n" + bcolors.ENDC)
    print(printout.to_string())
    print("\n" + bcolors.OKBLUE + "Importing data " + file_path + " ..." + bcolors.ENDC)

    # ch_1 = data[0] ch_2 = data[1] ch_3 = data[2] ch_4 = data[3]

    for channel in data:
        ch_data = data[channel]
        print(ch_data.head(n=5).to_string())

        # Plot the signal
        fig, (ax1, ax2) = plot.subplots(2, 1)
        fig.suptitle('Spectrogram of ' + save_name + "Channel: " + str(channel))

        # Time domain
        ax1.plot(ch_data - np.mean(ch_data))
        ax1.set(xlabel='Sample', ylabel='Amplitude')

        # Frequency domain
        # Pxx, freqs, bins, im = ax2.specgram(ch_data - np.mean(ch_data), Fs=samplingFrequency, window=signal.get_window('hamming', 128))
        # Pxx, freqs, bins, im = ax2.specgram(ch_data - np.mean(ch_data), Fs=samplingFrequency, window=signal.get_window(('kaiser', 8.0), 256))
        # Pxx, freqs, bins, im = ax2.specgram(ch_data - np.mean(ch_data), Fs=samplingFrequency, NFFT=1024, noverlap=512,
        #                                     window=signal.get_window('hamming', 1024))
        Pxx, freqs, bins, im = ax2.specgram(ch_data - np.mean(ch_data), Fs=samplingFrequency, NFFT=512, noverlap=500)

        # - Pxx: the periodogram
        # - freqs: the frequency vector
        # - bins: the centers of the time bins
        # - im: the matplotlib.image.AxesImage instance representing the data in the plot

        # Velocity from frequency
        velocity = freqs / 2 * c / freq_tx

        # Try that one:
        plot_data = 10 * np.log10(Pxx)
        spec = ax2.pcolormesh(bins, freqs, plot_data, vmin=np.percentile(plot_data, 20), vmax=np.percentile(plot_data, 100),
                          cmap='jet')  # cmap='plasma'
        ax2.axis('tight')
        ax2.set(xlabel='Time', ylabel='Frequency', ylim=(0, 20000))
        ax2_divider = make_axes_locatable(ax2)
        cax1 = ax2_divider.append_axes("right", size="7%", pad="2%")
        cb1 = colorbar(spec, cax=cax1)

        # Adjust height
        fig.subplots_adjust(hspace=0.3)

        # Save and close plot
        save_path = os.path.splitext(file_path)[0]
        plot.savefig(save_path + 'ch_' + str(channel) + '.png')
        # print("Saved as " + save_name + '_new.png')
        plot.close(fig=fig)
        print("Spectra of " + save_name + ", Channel " + str(channel) + " has been saved.")

# %% Finish program
print("Done. Spectra of " + str(len(file_paths)) + " datasets have been saved.")
