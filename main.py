import os 
import numpy as np 
import matplotlib.pyplot as plt 

from scipy import signal as signal
from scipy.io import wavfile

import settings as sett

paths = sett.paths()
params = sett.parameters()

#Extract data
data_path = '../Samples/'
samplerate, sample = wavfile.read(os.path.join(data_path, 'example01.wav'))

# Remove stereo
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



def create_spatial_masks(size, res, random=False):
	if random:
		spatial_tensor = np.random.rand(size, size, res)
	else:
		spatial_tensor = np.zeros((size, size, res))

	return spatial_tensor


def create_amplitude_mask(res, random=False):
	if random:
		amplitude_mask = np.random.rand(res)
	else:
		amplitude_mask = np.ones(res)

	return amplitude_mask


frequencies, times, spectrogram = spectro(sample, samplerate, plot=False)

spatial_tensor = create_spatial_masks(params.size_implant, params.freq_resolution, random=True)
amplitude_mask = create_amplitude_mask(params.freq_resolution, random=True)



