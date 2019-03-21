#%% import the libraries
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
freq_tx = 77000                 # in Hz
samplingFrequency = 10001 / 5   # Sampling Frequency

# %% Constants
c = 3 * 10**8

# %% Open file select window
root = tk.Tk()
root.withdraw()
file_paths = filedialog.askopenfilenames(parent=root, title='Choose datasets')

for file_path in file_paths:

    # Read data file with pandas
    data = pd.read_csv(file_path, skiprows=2)
    save_name = os.path.basename(file_path)

    # Print preview
    printout = data.head(n=5)
    print("\n" + bcolors.OKGREEN + "Preview:\n" + bcolors.ENDC)
    print(printout.to_string())
    print("\n" + bcolors.OKBLUE + "Importing data " + file_path + " ..." + bcolors.ENDC)

    cpx = []
    ampl = []

    for index, row in data.iterrows():

        re = row[data.columns.values[1]]
        im = row[data.columns.values[2]]

        cpx_num = complex(re, im)
        ampl_num = math.sqrt(re*re + im*im)
        cpx.append(cpx_num)
        ampl.append(ampl_num)

    signalData = cpx

    # Plot the signal
    fig, (ax1, ax2) = plot.subplots(2, 1)
    fig.suptitle('Spectrogram of ' + save_name)
    
    # Time domain
    ax1.plot(ampl - np.mean(ampl))
    ax1.set(xlabel='Sample', ylabel='Amplitude')
    
    # Frequency domain
    # Pxx, freqs, bins, im = ax2.specgram(signalData - np.mean(signalData), Fs=samplingFrequency, window=signal.get_window('hamming',256))
    # Pxx, freqs, bins, im = ax2.specgram(signalData - np.mean(signalData), Fs=samplingFrequency, window=signal.get_window(('kaiser', 8.0), 256))
    Pxx, freqs, bins, im = ax2.specgram(signalData - np.mean(signalData), Fs=samplingFrequency, NFFT=128, noverlap=64 , 
                                        window=signal.get_window('hamming', 128))
    # - Pxx: the periodogram
    # - freqs: the frequency vector
    # - bins: the centers of the time bins
    # - im: the matplotlib.image.AxesImage instance representing the data in the plot

    # Velocity from frequency
    velocity = freqs / 2 * c / freq_tx

    # Try that one:
    plot_data = 10 * np.log10(Pxx)
    spec = ax2.pcolormesh(bins, freqs, plot_data, vmin=np.percentile(plot_data, 85), vmax=np.percentile(plot_data, 100), cmap='jet')     # cmap='plasma'
    ax2.axis('tight')
    ax2.set(xlabel='Time', ylabel='Frequency', ylim=(-200, 400))
    ax2_divider = make_axes_locatable(ax2)
    cax1 = ax2_divider.append_axes("right", size="7%", pad="2%")
    cb1 = colorbar(spec, cax=cax1)

    # im.set_clim(-110, -80)
    # im.set_clim(np.percentile(im.get_array(), 25), np.max(im.get_array()))
    # ax2.set(xlabel='Time', ylabel='Frequency')
    # ax2.set_yscale('symlog')
    # ax2_divider = make_axes_locatable(ax2)
    # cax1 = ax2_divider.append_axes("right", size="7%", pad="2%")
    # cb1 = colorbar(im, cax=cax1)
    
    # Adjust height
    fig.subplots_adjust(hspace=0.3)
    
    # Save and close plot    
    save_path = os.path.splitext(file_path)[0]
    plot.savefig(save_path + '.png')
    # print("Saved as " + save_name + '_new.png')
    plot.close(fig=fig)
    print("Spectra of " + save_name + " has been saved.")



# %% Finish program
print("Done. Spectra of " + str(len(file_paths)) + " datasets have been saved.")


# %% Notes

    # plot.subplot(211)
    # plot.title('Spectrogram of ' + file_path)
    # plot.plot(ampl - np.mean(ampl))
    # plot.xlabel('Sample')
    # plot.ylabel('Amplitude')

    # plot.subplot(212)
    # Pxx, freqs, bins, im = plot.specgram(signalData - np.mean(signalData), Fs=samplingFrequency)
    # plot.xlabel('Time')
    # plot.ylabel('Frequency')
    # plot.tight_layout()
    # plot.colorbar()
    # plot.clim(-110, -80)
    # plot.show()