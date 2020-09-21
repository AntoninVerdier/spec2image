import os 
import numpy as np 
import matplotlib.pyplot as plt 

from scipy import signal as signal
from scipy.io import wavfile

#Extract data
data_path = '../Samples/'
samplerate, sample = wavfile.read(os.path.join(data_path, 'example01.wav'))
sample = np.transpose(sample[:, 0])


def spectro(sample, samplerate, plot=False):
	frequencies, times, spectrogram = signal.spectrogram(sample, samplerate)

	if plot:
		# To save in desired location
		plt.pcolormesh(times, frequencies, spectrogram)
		plt.ylabel('Frequency [Hz]')
		plt.xlabel('Time [sec]')
		plt.show()

	return frequencies, times, spectrogram


frequencies, times, spectrogram = spectro(sample, samplerate, plot=True)

