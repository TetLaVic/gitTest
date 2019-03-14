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
file_path = filedialog.askopenfilename()

# Read data file with pandas
data = pd.read_csv(file_path, skiprows=2)

# Print preview
printout = data.head(n=5)
print("\n" + bcolors.OKGREEN + "Preview:\n" + bcolors.ENDC)
print(printout.to_string())
print("\n" + bcolors.OKBLUE + "Importing data ..." + bcolors.ENDC)


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

plot.subplot(211)
plot.title('Spectrogram of ' + file_path)
plot.plot(ampl - np.mean(ampl))
plot.xlabel('Sample')
plot.ylabel('Amplitude')

plot.subplot(212)
plot.specgram(signalData - np.mean(signalData), Fs=samplingFrequency)
plot.xlabel('Time')
plot.ylabel('Frequency')
plot.colorbar()
plot.clim(-110, -80)
# plot.show()

save_name = os.path.splitext(file_path)[0]

plot.savefig(save_name + '2.png')
print("Saved as " + save_name + '2.png')

