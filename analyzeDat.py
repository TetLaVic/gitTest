# %%
# Input:    data file with measurement data where the column vector represents 
#           the channel. No I & Q (= not complex).
#
# date:     02.04.2019
# author:   Patrick Rippl


# %% import the libraries
from classes.format import bcolors
# from classes.my_filters import cannyEdgeDetector
# from Pillow import Image
# from skimage import feature

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
    print("\n" + bcolors.OKBLUE + "Importing data " + file_path + " ..." + bcolors.ENDC)
    data = pd.read_csv(file_path, header=None, sep="\s+")
    
    #---- Preprocessing
    # Downsampling to improve calculation time
    downsample_factor = 1
    downsampled = data.iloc[::downsample_factor, :]

    preproc_data = downsampled
    preproc_data = preproc_data - preproc_data.mean()   # Substract DC from each channel
    
    # Check if IQ (number of channels == 2)
    if preproc_data.shape[1] == 2:
        preproc_data['complex'] = preproc_data[0] + preproc_data[1] * 1j
        preproc_data['delta'] = preproc_data[0] - preproc_data[1]
        print("\n Data with I&Q. \n")
    else:
        preproc_data['average'] = preproc_data.mean(numeric_only=True, axis=1)  # Add column with row average for all recorded channels
        preproc_data['median'] = preproc_data.median(numeric_only=True, axis=1) # Add column with row median  for all recorded channels
    # preproc_data['sum'] = data.sum(numeric_only=True, axis=1)       # Add column with row  sum    for all recorded channels  
        print("\n Data without I&Q. \n")
        
    # Define Save paths
    save_name = os.path.basename(file_path)
    save_path = os.path.splitext(file_path)[0]
    
    # Print preview
    printout = preproc_data.head(n=5)
    print("Done. \nPreview:\n" + printout.to_string() + "\n")
    
    # Save signal time domain chart
    plot.plot(preproc_data)
    plot.legend(preproc_data)
    plot.savefig(save_path + '_Signals' + '.png')
    plot.close()
    print(bcolors.OKGREEN + "\nTime domain of " + save_name + " has been saved." + bcolors.ENDC)

    for channel in preproc_data:
        ch_data = preproc_data[channel]     # print(ch_data.head(n=5).to_string())

        # Plot the signal
        # fig, (ax1, ax2, ax3) = plot.subplots(3, 1)
        fig, (ax1, ax2, ax3) = plot.subplots(3, 1)
        fig.suptitle('Spectrogram of ' + save_name + "Channel: " + str(channel))

        # ax1: Time domain
        ax1.plot(ch_data)   # Add "- np.mean(ch_data)", if data has DC offset
        ax1.set(xlabel='Sample', ylabel='Amplitude')

        # ax2: Frequency domain
        Pxx, freqs, bins, im = ax2.specgram(ch_data, Fs=samplingFrequency/downsample_factor,
                                            NFFT=2048, noverlap=512, window=signal.get_window('hann', 2048))

        # - Pxx: the periodogram
        # - freqs: the frequency vector
        # - bins: the centers of the time bins
        # - im: the matplotlib.image.AxesImage instance representing the data in the plot

        # Velocity from frequency
        velocity = (freqs*c) / (2*freq_tx)

        # Try that one:
        plot_data = 10 * np.log10(Pxx)
        
        # Post Processing:
        # 1)
        #plot_data = signal.medfilt2d(plot_data, 3)
        
        # 2a)
        # img = Image.fromarray(np.uint8(plot_data) , 'L')
        # TODO: my_canny
        # detector = cannyEdgeDetector(plot_data, sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.17, weak_pixel=100)
        # plot_data = detector.detect()
        # plot_data = np.asarray(plot_data)
        
        # 2b)
        # plot_data = feature.canny(plot_data, low_threshold  = 0.30 * np.max(plot_data),
        #                           high_threshold=0.40 * np.max(plot_data))
        

        # Plot Spectrum
        spec = ax2.pcolormesh(bins, freqs, plot_data, cmap='jet',
                              vmin=np.percentile(plot_data, 95),
                              vmax=np.percentile(plot_data, 100))
        ax2.axis('tight')
        ax2.set(xlabel='Time [s]', ylabel='Frequency [Hz]', ylim=(0, samplingFrequency/(2*2*downsample_factor*10)))
        ax2_divider = make_axes_locatable(ax2)
        cax1 = ax2_divider.append_axes("right", size="7%", pad="2%")
        cb1 = colorbar(spec, cax=cax1)
        
        
        #fft2FromSpec = np.fft.fft2(Pxx)
        #ax3.plot(fft2FromSpec)
        
        
        #FS = np.fft.fftn(Pxx)
        #ax3.imshow(np.log(np.abs(np.fft.fftshift(FS))**2))

        FS = np.fft.fftn(plot_data)
        fft2plt = ax3.imshow(np.log(np.abs(np.fft.fftshift(FS))**2),
                             vmin=np.percentile(np.log(np.abs(np.fft.fftshift(FS))**2), 80),
                             vmax=np.percentile(np.log(np.abs(np.fft.fftshift(FS))**2), 100))
        ax3.axis('tight')
        ax3_divider = make_axes_locatable(ax3)
        cax2 = ax3_divider.append_axes("right", size="7%", pad="2%")
        cb2 = colorbar(fft2plt, cax=cax2)
        ax3.set(xlim = (FS.shape[1]/2, FS.shape[1]/2 + 0.25*FS.shape[1]/2))     #ylim = (FS.shape[0]/2, FS.shape[0]/2 + 0.25*FS.shape[1]/2))
        fig.show()
        
        # [X, Y] = np.meshgrid(Pxx)
        # S = np.sin(X) + np.cos(Y) + np.random.uniform(0, 1, X.shape)
        # FS = np.fft.fftn(S)
        # ax3.imshow(np.log(np.abs(np.fft.fftshift(FS))**2))        
        

        # Keep only top 15% of occurring freqs
        # plot_data_modified = plot_data.flatten()
        # plot_data_modified = plot_data_modified[plot_data_modified > np.percentile(plot_data_modified, 85)]

        # ax3.plot(histData[1], histData[0])
        # ax3.hist(plot_data_modified, bins=100)
        # ax3.set(xlabel='Pxx [dB]', ylabel='#', xlim=(np.min(plot_data), np.max(plot_data)))
        # TODO: Global np.min(plot_data) and np.max(plot_data) from Channel 1..n to make results comparable
        
        fig2, (ax1, ax2, ax3) = plot.subplots(3, 1)
        
        Pxx, freqs, im = ax1.magnitude_spectrum(ch_data, Fs=samplingFrequency/downsample_factor,
                                            Fc=0, pad_to=None, sides='default')

        Pxx, freqs, im = ax2.angle_spectrum(ch_data, Fs=samplingFrequency/downsample_factor,
                                            Fc=0, pad_to=None, sides='default')

        Pxx, freqs, im = ax3.phase_spectrum(ch_data, Fs=samplingFrequency/downsample_factor,
                                            Fc=0, pad_to=None, sides='default')
        
        # FFT and FFT2
        fig3, (ax1, ax2) = plot.subplots(2, 1)
        fft1 = np.fft.fft(ch_data)
        N = int(len(ch_data)/2+1)
        X = np.linspace(0, samplingFrequency/2, N, endpoint=True)
        # Applying Hanning Window
        hann = np.hanning(len(ch_data))
        Yhann = np.fft.fft(hann * ch_data)
        
        # fft2 = np.fft.fft2(ch_data)
        ax1.plot(X, 2*np.abs(Yhann[:N])/N)
        ax1.set(title  = "Amplitude spectrum",
                xlabel = "Frequency ($Hz$)",
                ylabel = "Amplitude ($Volt$)",
                xlim = (0,2500))
        
        Y2 = np.fft.ifft(Yhann[:2500])
        ax2.plot(Y2)
        
        # Adjust height
        fig.subplots_adjust(hspace=0.75)
        fig2.subplots_adjust(hspace=0.75)
        fig3.subplots_adjust(hspace=1)
        
        # Save and close plot
        plot.figure(fig.number)
        plot.savefig(save_path + 'ch_' + str(channel) + '.png', format='png')   # dpi=900
        plot.close(fig=fig)
        
        plot.figure(fig2.number)
        plot.savefig(save_path + 'ch_' + str(channel) + '_2.png')
        plot.close(fig=fig2)
        
        plot.figure(fig3.number)
        plot.savefig(save_path + 'ch_' + str(channel) + '_3.png')  
        plot.close(fig=fig3)
        
        # print("Saved as " + save_name + '_new.png')

        print(bcolors.OKGREEN + "\nSpectrum of " + save_name + ", Channel " + str(channel)
              + " has been saved." + bcolors.ENDC)

# %% Finish program
print(bcolors.OKBLUE + "\nFinished. Spectra of " + str(len(file_paths)) + " datasets have been saved." + bcolors.ENDC)
