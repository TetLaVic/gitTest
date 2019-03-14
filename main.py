# import the libraries
from classes.format import bcolors
import matplotlib.pyplot as plot
import pandas as pd
import os
import numpy as np
import math
import tkinter as tk
from tkinter import filedialog


# Open file select window
root = tk.Tk()
root.withdraw()
file_paths = filedialog.askopenfilenames(parent=root, title='Choose a dataset')

for file_path in file_paths:

    # Read data file with pandas
    data = pd.read_csv(file_path, skiprows=2)

    # Print preview
    printout = data.head(n=5)
    print("\n" + bcolors.OKGREEN + "Preview:\n" + bcolors.ENDC)
    print(printout.to_string())
    print("\n" + bcolors.OKBLUE + "Importing data " + file_path + " ..." + bcolors.ENDC)


    # Sampling Frequency
    samplingFrequency = 60000/8

    cpx = []
    ampl = []

    for index, row in data.iterrows():

        re = row['re:Trc1_S22']
        im = row['im:Trc1_S22']

        cpx_num = complex(re, im)
        ampl_num = math.sqrt(re*re + im*im)
        cpx.append(cpx_num)
        ampl.append(ampl_num)

    signalData = cpx

    # Plot the signal read from wav file
    # fig, (ax1, ax2) = plot.subplots(2, 1)
    # ax1.set_title('Spectrogram of ' + file_path)
    # ax1.plot(ampl - np.mean(ampl))
    # ax1.set(xlabel='Sample', ylabel='Amplitude')
    #
    # Pxx, freqs, bins, im = ax2.specgram(signalData - np.mean(signalData), Fs=samplingFrequency)
    # ax2.set(xlabel='Time', ylabel='Frequency')
    # # ax = plot.gca()  # get the current axes
    # PCM = ax2.get_children()[2]  # get the mappable, the 1st and the 2nd are the x and y axes
    # fig.colorbar([], PCM)
    # fig.clim(-110, -80)

    # - Pxx: the periodogram
    # - freqs: the frequency vector
    # - bins: the centers of the time bins
    # - im: the matplotlib.image.AxesImage instance representing the data in the plot

    plot.subplot(211)
    plot.title('Spectrogram of ' + file_path)
    plot.plot(ampl - np.mean(ampl))
    plot.xlabel('Sample')
    plot.ylabel('Amplitude')

    plot.subplot(212)
    Pxx, freqs, bins, im = plot.specgram(signalData - np.mean(signalData), Fs=samplingFrequency)
    plot.xlabel('Time')
    plot.ylabel('Frequency')
    plot.tight_layout()
    plot.colorbar()
    plot.clim(-110, -80)
    # plot.show()

    save_name = os.path.splitext(file_path)[0]

    plot.savefig(save_name + '_new.png')
    print("Saved as " + save_name + '_new.png')

print("Done. Spectra of " + str(len(file_paths)) + " datasets have been saved.")
